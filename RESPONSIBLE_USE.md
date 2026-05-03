# Responsible Use

This repository ships a working ShadowLogic backdoor against a popular open
object detector. It is published, gated, and watermarked to support
HiddenLayer's defensive research mission. It is not a tool for offensive use.

## Authorized uses

- Red-team and adversarial-evaluation engagements where the model owner
  has explicitly authorized the use of supply-chain-style payloads.
- Federal / DoD AI security education, where decision makers need to see
  the threat in person to fund mitigations.
- Academic and industry research on detection, scanning, and runtime
  defenses for graph-level model backdoors.
- Internal blue-team training: scanning ML pipelines for ShadowLogic-style
  signatures, validating that mitigations work.

## Prohibited uses

- Deployment against any production system without the system owner's
  explicit written authorization.
- Insertion into a model distribution channel (model hub, internal model
  registry, fine-tune service) where downstream consumers cannot opt in.
- Removing the watermark `metadata_props` or the `/shadowlogic/...` node
  prefix to defeat detection.
- Targeting any individual or group on the basis of protected characteristics.

## Why we still publish it

The ShadowLogic technique is already documented publicly by HiddenLayer
research. Withholding a working reference implementation does not protect
defenders — it only forces every defender to reinvent it from the
disclosure. By gating the repository (Hugging Face manual approval) and
shipping with this `RESPONSIBLE_USE.md`, we prefer to put the artifact in
the hands of vetted defenders so they can build scanners and mitigations
against the real thing.

## Reporting concerns

If you believe this repository or its model has been misused, contact
HiddenLayer at https://hiddenlayer.com/contact/ or open a private
disclosure issue against this repository.
