// Invoke the installed, pinned CLI's own OAuth implementation without its UI.
// No model calls, custom OAuth endpoints, or credential logging.
import { readFile } from 'node:fs/promises';
import { setTimeout as delay } from 'node:timers/promises';

const runtime = new URL('../.venv/gemini_cli/', import.meta.url);
const pkg = JSON.parse(await readFile(new URL('node_modules/@google/gemini-cli/package.json', runtime), 'utf8'));
if (pkg.version !== '0.62.0') {
  throw new Error('Login helper requires Gemini CLI 0.62.0; inspect its exports before upgrading.');
}
const core = await import(new URL('node_modules/@google/gemini-cli/bundle/chunk-MLY4WQFO.js', runtime));
if (typeof core.getOauthClient !== 'function') throw new Error('Official OAuth entry point is unavailable.');
if (process.argv.includes('--check')) {
  console.log('Official Gemini CLI OAuth entry point loaded; no authentication or model request made.');
  process.exit(0);
}
if (!process.stdin.isTTY) throw new Error('Run login-only in your interactive server terminal.');
try {
  const client = await core.getOauthClient('oauth-personal', {
    getProxy: () => undefined,
    isBrowserLaunchSuppressed: () => true,
    isInteractive: () => true,
    getAcpMode: () => false,
  });
  const { token } = await client.getAccessToken();
  if (!token) throw new Error('Google did not return a usable access token.');
  await client.getTokenInfo(token);
  // The upstream token event persists credentials asynchronously.
  const credentialsPath = new URL('account/.gemini/oauth_creds.json', runtime);
  let saved = false;
  for (let attempt = 0; attempt < 20; attempt++) {
    try {
      const credentials = JSON.parse(await readFile(credentialsPath, 'utf8'));
      saved = Boolean(credentials.refresh_token);
    } catch { /* Wait for the official credential writer. */ }
    if (saved) break;
    await delay(250);
  }
  if (!saved) throw new Error('OAuth returned, but reusable credentials were not saved.');
  console.log('Google OAuth login verified and credentials saved.');
  console.log('No model calls made. Model access and subscription entitlement still need checking.');
  process.exit(0);
} catch (error) {
  // Do not serialize provider response objects, which can contain credentials.
  console.error('Google OAuth login did not complete. See the authentication error above.');
  console.error('Error type:', error?.name || 'Error');
  const message = String(error?.message || '').replace(/https?:\/\/\S+/g, '[URL hidden]')
    .replace(/[A-Za-z0-9_./+\-=]{32,}/g, '[long value hidden]');
  if (message) console.error('Reason:', message.slice(0, 700));
  process.exit(1);
}
