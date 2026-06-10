# chang-han-chen.github.io

Personal website of Chang-Han Chen, built with a minimal custom Jekyll setup (no theme).

## Structure

- `_layouts/` — three layouts: `default.html` (nav + head), `page.html`, `post.html`
- `assets/css/site.css` — the only stylesheet
- `index.md` — About (front page)
- `publications/`, `resume/`, `blog/` — the other tabs
- `_posts/` — blog posts in Markdown (kramdown + MathJax; add a `<details class="post-toc">` block for a collapsible table of contents)

## Local development

```
bundle install
bundle exec jekyll serve
```

Then open http://localhost:4000. GitHub Pages builds the site automatically on push.
