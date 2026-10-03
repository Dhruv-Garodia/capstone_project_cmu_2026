# Colab Pro over SSH — setup

## 1. Upload the data (once)

Drag `colab_bundle/Eponge/` into Google Drive so it becomes `MyDrive/Eponge/` (about 2.5 GB).

## 2. In a Colab notebook (GPU runtime, Colab Pro)

Paste your public key into `PUBKEY` (on the laptop: `cat ~/.ssh/id_ed25519.pub`), then run this cell:

```python
PUBKEY = "ssh-ed25519 AAAA...  your-key-comment"

import time
!apt-get -qq update && apt-get -qq install -y openssh-server > /dev/null
!mkdir -p /var/run/sshd /root/.ssh && chmod 700 /root/.ssh
open("/root/.ssh/authorized_keys", "w").write(PUBKEY + "\n")
!chmod 600 /root/.ssh/authorized_keys
!sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/; s/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
!env | grep -E '^(PATH|LD_LIBRARY_PATH|CUDA|NVIDIA)' | sed 's/^/export /' >> /root/.bashrc
!/usr/sbin/sshd
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared
from google.colab import drive
drive.mount("/content/drive")
!nohup cloudflared tunnel --url ssh://localhost:22 > /content/cloudflared.log 2>&1 &
time.sleep(8)
!grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /content/cloudflared.log | head -1
```

The last line prints the tunnel hostname, e.g. `https://quiet-river-1234.trycloudflare.com`.
Keep this notebook tab open; the tunnel dies with the runtime.

## 3. On the laptop

```bash
brew install cloudflared
```

Add to `~/.ssh/config` (replace the hostname each time you start a new runtime):

```
Host colab
  HostName quiet-river-1234.trycloudflare.com
  User root
  ProxyCommand cloudflared access ssh --hostname %h
  IdentityFile ~/.ssh/id_ed25519
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
```

Test: `ssh colab nvidia-smi`

## 4. Run the experiments

```bash
ssh colab 'cd /content/drive/MyDrive/Eponge && nohup sh run_experiments.sh > run.log 2>&1 &'
ssh colab 'tail -5 /content/drive/MyDrive/Eponge/run.log'
```

Everything is written under `MyDrive/Eponge/outputs/`, so a runtime reset loses nothing finished.
Rerunning the script skips models that already have `best.pt`.

## 5. Bring results back

```bash
rsync -av --exclude preprocessed.npy colab:/content/drive/MyDrive/Eponge/outputs/ ~/Desktop/CMU_life/Capstone/outputs/
```
