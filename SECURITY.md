# Security policy

## Reporting a vulnerability

Report a suspected vulnerability privately, through GitHub private vulnerability
reporting on this repository ("Report a vulnerability" under the Security tab).
Do not open a public issue, and do not include credentials, evaluator answers or a
working exploit in any public channel.

Reports we especially want:

- a way for agent-written code to leave the executor sandbox (`executor/`), reach
  the network or the host, or read files it was not given;
- a way to obtain evaluator references, withheld answers or grading keys from the
  participant materials;
- a way to make the verifier accept a score, replay or release that it should
  reject;
- a credential or evaluator answer committed to this repository.

We will acknowledge your report, keep you informed while we investigate, and agree
a disclosure date with you once a fix or mitigation is available.

## Scope

The benchmark runs untrusted, model-written code. Run it only inside the provided
executor container, on a machine that holds no credentials beyond the provider key
the run needs. A checksum proves byte identity, not that an artifact is safe.
