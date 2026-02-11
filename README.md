# amass
attention, memory, and aging study at stanford (AMASS)

## Prerequisites

### SSH Connection Multiplexing (Sherlock)

The `fetch_subject_data.sh` script transfers many files from Sherlock via SCP. To avoid re-authenticating with MFA for every file, add the following to your `~/.ssh/config`:

```
Host sherlock
    HostName login.sherlock.stanford.edu
    User <your-sunet-id>
    ControlMaster auto
    ControlPath ~/.ssh/control-%r@%h:%p
    ControlPersist 10m
```

This opens a persistent SSH connection on first login that is reused by all subsequent SCP calls. The connection stays alive for 10 minutes after the last transfer.
