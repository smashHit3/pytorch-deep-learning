---
alwaysApply: true
scene: git_message
---

# Git Commit Message Style Guide

## Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

## Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `style` | Code style (formatting, whitespace) |
| `refactor` | Code refactoring (no feature/fix) |
| `perf` | Performance improvement |
| `test` | Adding or modifying tests |
| `chore` | Build, config, or tooling changes |
| `ci` | CI/CD pipeline changes |

## Rules

1. **Language**
   - Always use **English** for all commit messages
   - No Chinese or other languages in subject, body, or footer

2. **Subject line**
   - Use imperative mood: "Add feature" not "Added feature"
   - No capitalization at start
   - No period at end
   - Max 50 characters

3. **Body** (optional)
   - Explain **what** and **why**, not **how**
   - Separate from subject with blank line
   - Each line max 72 characters

4. **Footer** (optional)
   - Reference issues: `Closes #123`
   - Breaking changes: `BREAKING CHANGE: description`

## Examples

```
feat(models): add mobilenet implementation

Add MobileNetV1 with depthwise separable convolutions.
Supports width multiplier for model scaling.

Closes #42
```

```
fix(train): correct learning rate scheduler step

Scheduler was stepping before epoch end, causing
premature LR decay. Now steps after each epoch.
```

```
docs(cv_sources): add README with usage examples
```

```
refactor(resnet): extract common block logic into base class
```