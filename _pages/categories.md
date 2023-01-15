---
layout: page
permalink: /topics/
title: Topics
---


<div id="archives">
{% assign category_order = "NLP, Reinforcement Learning, Deep Learning, Transformer, RNN, Data Science, Plan and Control, Graph Theory" | split: ", " %}
{% for category in site.categories | sort: category_order %}
  <div class="archive-group">
    {% capture category_name %}{{ category | first }}{% endcapture %}
    <div id="#{{ category_name | slugize }}"></div>
    <p></p>
    
    <h3 class="category-head">{{ category_name }}</h3>
    <a name="{{ category_name | slugize }}"></a>
    {% for post in site.categories[category_name] | sort: "year" %}
    <article class="archive-item">
      <h4><a href="{{ site.baseurl }}{{ post.url }}">{% if post.title and post.title != "" %}{{ post.title }} ({{ post.year }}){% else %}{{post.excerpt |strip_html}}{%endif%}</a></h4>
    </article>
    {% endfor %}
  </div>
{% endfor %}
</div>