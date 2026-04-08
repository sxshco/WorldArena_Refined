# Public Release Notes

This copy is intended for publishing to a public Git repository.

What was sanitized:
- removed repository history and local experiment outputs
- replaced hard-coded personal paths with relative paths or generic placeholders
- switched default cache/output locations to repo-local paths
- added `.gitignore` entries for generated data, model weights, caches, and private outputs

Before publishing:
- review `video_quality/config/config.yaml` and point checkpoints to your local model files
- review `embodied_task/policy_conf/*.yaml` and set dataset/model paths for your environment
- keep secrets such as API keys in environment variables only
