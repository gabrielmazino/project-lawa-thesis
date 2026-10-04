-- =============================================================
-- Project LAWA: user profiles
--
-- Supabase Auth stores each user's email and bcrypt-hashed password
-- in auth.users. This table stores everything else about the user.
--
-- Users can log in with their EMAIL or their USERNAME. For username
-- logins the app first calls get_email_for_username() (section 6),
-- then signs in with that email. A real email also lets users reset
-- a forgotten password themselves.
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
-- Usernames: lowercase letters, numbers, dot, underscore, hyphen.
-- No "@", so the login page can tell a username from an email.
create table public.profiles (
  id          uuid primary key references auth.users (id) on delete cascade,
  username    text not null unique check (username ~ '^[a-z0-9._-]{3,30}$'),
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
-- username: from user metadata if provided, otherwise built from the
--   part of the email before "@" (e.g. juan.cruz@gmail.com -> juan.cruz).
--   If that username is taken, a short suffix is added. An admin can
--   change it afterwards in the profiles table.
-- full_name: from user metadata if provided, otherwise the username.
-- role: read from app metadata (which users cannot edit themselves),
--   never from user metadata. Defaults to environmental_officer.
create function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
declare
  v_username text;
begin
  v_username := coalesce(
    nullif(trim(new.raw_user_meta_data ->> 'username'), ''),
    split_part(new.email, '@', 1)
  );
  v_username := left(regexp_replace(lower(v_username), '[^a-z0-9._-]', '', 'g'), 25);
  if length(v_username) < 3 then
    v_username := 'user';
  end if;
  if exists (select 1 from public.profiles where username = v_username) then
    v_username := v_username || '-' || left(replace(new.id::text, '-', ''), 4);
  end if;

  insert into public.profiles (id, username, full_name, role)
  values (
    new.id,
    v_username,
    coalesce(nullif(trim(new.raw_user_meta_data ->> 'full_name'), ''), v_username),
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


-- 6. Username login: find the email that belongs to a username ---
-- Called by the login page before the user is signed in, so it must
-- be available to anon. Returns null for unknown or inactive users.
-- Known limitation: someone who guesses a valid username can learn
-- that user's email address.
create function public.get_email_for_username(p_username text)
returns text
language sql
stable
security definer
set search_path = ''
as $$
  select u.email
  from public.profiles p
  join auth.users u on u.id = p.id
  where p.username = lower(trim(p_username))
    and p.is_active
  limit 1;
$$;

revoke execute on function public.get_email_for_username(text) from public;
grant execute on function public.get_email_for_username(text) to anon, authenticated;


-- 7. Security (Row Level Security) -------------------------------
alter table public.profiles enable row level security;

-- "Automatically expose new tables" is off, so access is granted
-- explicitly. Logged-out visitors (anon) cannot read the table.
grant select on public.profiles to authenticated;
grant update (username, full_name, role, is_active) on public.profiles to authenticated;
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

-- Only system admins can change profiles (username, name, role,
-- active status). Regular users cannot edit their own role.
create policy "System admins can update profiles"
  on public.profiles for update
  to authenticated
  using ((select public.is_system_admin()))
  with check ((select public.is_system_admin()));
