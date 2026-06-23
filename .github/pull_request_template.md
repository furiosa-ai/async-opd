## Summary

Briefly describe what changed, why it is needed, and any release, privacy, or
dependency impact reviewers should know about.

## Type of change

- [ ] Documentation only
- [ ] Bug fix
- [ ] Feature
- [ ] Refactor/cleanup
- [ ] Testing/CI
- [ ] Other

## Validation

- [ ] `python -m pytest tests/test_cli_entrypoints.py tests/test_public_api.py tests/test_cpu_stub_pipeline.py -q -o addopts=""`
- [ ] Targeted tests relevant to this change:

```text
<commands and results>
```

## Release/privacy checklist

- [ ] I did not add secrets, private hostnames/IPs, local paths, customer names, or internal-only docs.
- [ ] Public docs/commands still work or are covered by tests.
- [ ] Dependency versions were not changed, or dependency changes are explicitly justified.
- [ ] If this changes release packaging/docs/dependencies, I refreshed or noted the research-release check output.
