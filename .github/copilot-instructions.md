# Copilot Instructions for karray.github.io

Personal blog and portfolio site built with **Jekyll** using the **Minima theme** and hosted on GitHub Pages.

## Architecture Overview

- **Static Site Generator**: Jekyll with `github-pages` gem (~232) for GitHub Pages compatibility
- **Theme**: Minima 2.5 with extensive customization via `_sass/` partials
- **Comments**: Staticman (hosted on Heroku) stores comments as YAML in `_data/comments/`
- **Math Rendering**: MathJax (configured in [\_layouts/post.html](_layouts/post.html)) - use `$...$` for inline, `$$...$$` for blocks
- **Analytics**: Google Analytics (G-GNT1D799WL)

## Key Directories

| Directory    | Purpose                                                                   |
| ------------ | ------------------------------------------------------------------------- |
| `_posts/`    | Blog posts in Markdown with YAML front matter                             |
| `_layouts/`  | Custom layouts: `home.html`, `post.html`, `about.html`                    |
| `_includes/` | Reusable partials (e.g., `comments.html`, `reference.html` for citations) |
| `_sass/`     | SCSS partials imported via [assets/main.scss](assets/main.scss)           |
| `_data/`     | JSON data files for reputation widgets and comments storage               |
| `examples/`  | Standalone HTML demos (LaFAM, StyleGAN) with their own JS/CSS             |
| `_plugins/`  | Custom Liquid tags (e.g., `WakaTag.rb` for coding stats)                  |

## Development Workflow

```bash
# Install dependencies
bundle install

# Serve locally with live reload (restart if _config.yml changes)
bundle exec jekyll serve

# Build for production
bundle exec jekyll build
```

## Content Patterns

### Blog Post Front Matter

```yaml
---
layout: post
title: Your Post Title
date: 2025-01-01 00:00:00 +0200
tags: Tag1, Tag2, Tag3
description: Brief description for excerpts and SEO.
---
```

### Excerpt Handling

Use `<!--more-->` to mark excerpt end (configured in `_config.yml` as `excerpt_separator`).

### Academic References

Use the custom reference system in posts:

```html
<!-- In-text citation -->
<a href="#authorYYYYfirstword" data-ref="authorYYYYfirstword">Author et al.</a>

<!-- Reference list at post end (uses _includes/reference.html) -->
{% include reference.html ref="Author, A.|2024|Paper Title" %}
```

The anchor format is: `{lastName}{year}{firstTitleWord}` slugified.

### Dynamic Header Behavior

Posts use a shrinking header that becomes fixed on scroll. CSS classes: `.dynamic-header`, `.fixed-header`, `.hidden-header`. Logic in [assets/js/posts.js](assets/js/posts.js).

## Interactive Demos (`examples/`)

The `examples/lafam/` directory contains a browser-based ML demo:

- Uses ONNX Runtime (`worker.js`) for model inference
- Service worker for offline caching
- Canvas-based visualization with heatmap overlays
- Standalone HTML - doesn't use Jekyll layouts

## Styling Guidelines

- Import order in `main.scss`: base minima → custom partials
- Breakpoint variable: `$brakepoint: 350px` (note: typo preserved for compatibility)
- Header animations use `.light` divs for floating effect (see `header_background.scss`)

## Staticman Comments

Comments flow: Form → Heroku endpoint → YAML files in `_data/comments/{slug}/`
Configuration in [staticman.yml](staticman.yml). Currently auto-approved (moderation: false).
