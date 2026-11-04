#!/bin/sh
set -e
# No BETTER_AUTH_SECRET given: generate one on first boot and keep it in the data volume, so sessions survive restarts
# and no shared default secret ever exists.
if [ -z "$BETTER_AUTH_SECRET" ]; then
  [ -s /data/.auth-secret ] || head -c 32 /dev/urandom | base64 > /data/.auth-secret
  BETTER_AUTH_SECRET=$(cat /data/.auth-secret)
  export BETTER_AUTH_SECRET
fi
exec ./server
