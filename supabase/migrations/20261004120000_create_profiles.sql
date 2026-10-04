-- =============================================================
-- Project LAWA: user profiles
--
-- Supabase Auth stores each user's login and bcrypt-hashed password
-- in auth.users. This table stores everything else about the user.
--
-- Users log in with a USERNAME, not an email. Supabase Auth requires
-- an email, so every account uses an internal address:
--   username "jdelacruz"  ->  email "jdelacruz@lawa.local"
-- .local addresses never receive mail. The username is taken from
-- that email automatically and cannot be edited afterwards.
--
-- How to run: Supabase dashboard -> SQL Editor -> New query ->
-- paste this whole file -> Run. Run it only once.
-- =============================================================


-- 1. Roles ----------------------------------------------------
create type public.user_role as enum (
  'environmental_officer',
  'system_admin',
  'director'
);


-- 2. Profiles table -------------------------------------------
create table public.profiles (
  id          uuid primary key references auth.users (id) on delete cascade,
  username    text not null unique check (username <> '' and username = lower(username)),
  full_name   text not null check (trim(full_name) <> ''),
  role        public.user_role not null default 'environmental_officer',
  is_active   boolean not null default true,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

comment on table public.profiles is 'Profile and role of each LAWA user. Login credentials live in auth.users.';


-- 3. Keep updated_at current ------------------------------------
create function public.set_updated_at()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger profiles_set_updated_at
  before update on public.profiles
  for each row execute function public.set_updated_at();


-- 4. Create a profile automatically for every new login account --
-- username is always the part of the email before "@", so it matches
-- what the user types on the login page.
-- full_name comes from user metadata when provided, else the username.
-- The role is read from app metadata (which users cannot edit
-- themselves), never from user metadata. Defaults to environmental_officer.
create function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
  insert into public.profiles (id, username, full_name, role)
  values (
    new.id,
    lower(split_part(new.email, '@', 1)),
    coalesce(new.raw_user_meta_data ->> 'full_name', split_part(new.email, '@', 1)),
    coalesce((new.raw_app_meta_data ->> 'role')::public.user_role, 'environmental_officer')
  );
  return new;
end;
$$;

revoke execute on function public.handle_new_user() from public, anon, authenticated;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();


-- 5. Helper: is the current user an active system admin? ---------
-- security definer lets it read profiles without triggering the RLS
-- policies below (which would otherwise call it again in a loop).
create function public.is_system_admin()
returns boolean
language sql
stable
security definer
set search_path = ''
as $$
  select exists (
    select 1
    from public.profiles
    where id = (select auth.uid())
      and role = 'system_admin'
      and is_active
  );
$$;

revoke execute on function public.is_system_admin() from public, anon;
grant execute on function public.is_system_admin() to authenticated;


-- 6. Security (Row Level Security) -------------------------------
alter table public.profiles enable row level security;

-- "Automatically expose new tables" is off, so access is granted
-- explicitly. Logged-out visitors (anon) get nothing.
-- username is not in the update list: changing it would break login.
grant select on public.profiles to authenticated;
grant update (full_name, role, is_active) on public.profiles to authenticated;
grant all on public.profiles to service_role;

-- Every logged-in user can read their own profile.
create policy "Users can read their own profile"
  on public.profiles for select
  to authenticated
  using (id = (select auth.uid()));

-- System admins can read every profile.
create policy "System admins can read all profiles"
  on public.profiles for select
  to authenticated
  using ((select public.is_system_admin()));

-- Only system admins can change profiles (names, roles, active status).
-- Regular users cannot edit their own role.
create policy "System admins can update profiles"
  on public.profiles for update
  to authenticated
  using ((select public.is_system_admin()))
  with check ((select public.is_system_admin()));
