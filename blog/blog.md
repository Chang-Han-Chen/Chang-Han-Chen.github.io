---
title: "Blog"
layout: page
permalink: /blog/
---

<ul class="post-list">
  {% for post in site.posts %}
  <li>
    <span class="post-date">{{ post.date | date: "%B %-d, %Y" }}</span>
    <a class="post-link" href="{{ post.url | relative_url }}">{{ post.title }}</a>
    {% if post.excerpt %}
    <p class="post-excerpt">{{ post.excerpt | strip_html | strip_newlines | truncate: 240 }}</p>
    {% endif %}
  </li>
  {% endfor %}
</ul>
