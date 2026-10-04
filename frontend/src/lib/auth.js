// Login logic. Passwords are checked by Supabase Auth, never by our code.
import { setRememberDevice, supabase } from './supabase'

const PROFILE_FIELDS = 'id, username, full_name, role, is_active'

const MESSAGES = {
  invalid: 'Incorrect email/username or password.',
  disabled: 'This account has been disabled. Contact your System Administrator.',
  unconfirmed: 'This email address has not been confirmed yet. Contact your System Administrator.',
  tooMany: 'Too many sign-in attempts. Please wait a few minutes and try again.',
  connection: 'Cannot reach the server. Check your internet connection and try again.',
}

function friendlyAuthError(error) {
  if (error.code === 'invalid_credentials') return MESSAGES.invalid
  if (error.code === 'email_not_confirmed') return MESSAGES.unconfirmed
  if (error.code === 'over_request_rate_limit') return MESSAGES.tooMany
  if (error.name === 'AuthRetryableFetchError') return MESSAGES.connection
  return error.message
}

// Turns what the user typed into an email. Anything without "@" is a username.
async function resolveEmail(identifier) {
  if (identifier.includes('@')) return identifier

  const { data: email, error } = await supabase.rpc('get_email_for_username', { p_username: identifier })
  if (error) throw new Error(MESSAGES.connection)
  // Same message as a wrong password, so we don't reveal which usernames exist.
  if (!email) throw new Error(MESSAGES.invalid)
  return email
}

export async function fetchProfile(userId) {
  const { data, error } = await supabase.from('profiles').select(PROFILE_FIELDS).eq('id', userId).maybeSingle()
  if (error) throw new Error(MESSAGES.connection)
  return data
}

// Signs in with an email or username. Returns the user's profile, or throws
// an Error whose message can be shown to the user.
export async function signIn(identifier, password, remember) {
  setRememberDevice(remember)

  const email = await resolveEmail(identifier.trim())
  const { data, error } = await supabase.auth.signInWithPassword({ email, password })
  if (error) throw new Error(friendlyAuthError(error))

  const profile = await fetchProfile(data.user.id).catch(async (err) => {
    await supabase.auth.signOut()
    throw err
  })
  if (!profile || !profile.is_active) {
    await supabase.auth.signOut()
    throw new Error(MESSAGES.disabled)
  }
  return profile
}

export async function signOut() {
  await supabase.auth.signOut()
}
