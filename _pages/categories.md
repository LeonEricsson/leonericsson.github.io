---
layout: page
permalink: /topics/
title: Topics
---


<div id="archives">
{% assign category_order = "Deep Learning, NLP, Reinforcement Learning, Transformer, Medical Imaging, RNN, Data Science, Plan and Control, Graph Theory" | split: ", " %}
{% for category in category_order %}
  <div class="archive-group">
      <div id="#{{ category | slugize }}"></div>
      <p></p>
      <h3 class="category-head">{{ category }}</h3>
      <a name="{{ category | slugize }}"></a>
    {% for post in site.categories[category]%}
    <article class="archive-item">
      <h4><a href="{{ site.baseurl }}{{ post.url }}">{% if post.title and post.title != "" %}{{ post.title }} ({{ post.year }}){% else %}{{post.excerpt |strip_html}}{%endif%}</a></h4>
    </article>
    {% endfor %}
  </div>
{% endfor %}
</div>