---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: page
title:  The Hitchiker's Guide to the ML Galaxy
permalink: /
---
# Latest posts
<ul>
{% for post in site.posts limit:3 %}
<li>
    <a href="{{ post.url }}">
    <h3> {{ post.title }} </h3>
    </a>
</li>
{% endfor %}
</ul>



# What is this blog?
It's a blog dedicated to AI and machine learning, which will predominantly be centred around research papers; a mix of older classics, forgotten treasures and the avant-garde.

Every week-ish, I will write a post as part of an ongoing series based on a broader topic. I plan on rotating series each week (so that we have some fun and variety), and I currently intend on having around 4-5 series on the lines of the list below:

* *CV* (Computer Vision): This series will cover papers framed around a broad set of topics in image classification, object detection, semantic image segmentation and many other CV-related subfields

* *NLP* (Natural Language Processing): Similar to CV, this series will focus on NLP, but will jump around different tasks and models (transformers, RNNs/LSTMs/GRUs/whatever the "in" flavor is these days, etc)

* *RL* (Reinforcement Learning): This one will cover a spectrum of RL-related algorithms (variants of policy gradients, q-learning, actor-critic, model-based) but will also go into areas that have intersections with RL, such as meta-learning, multi-task learning, etc.

* *Quests*: For all the other things going on in machine learning that don't quite fit into the first three categories, I'll add this series - as the name suggests, this will be a bit more exploratory than just paper reviews, going into all kinds of things, from AutoML to Deep Geometric ML and more! 

* *Foundations*: This series will go into the basics of machine learning; unlike the first four (which will be a bit more technical), this will be oriented towards any and all potential audiences - people interested in AI, people worried AI might kill their jobs, and even people who might not care now (but will hopefully care soon!). (P.S. This series is named after Isaac Asimov's Foundations series, a fantastic set of science-fiction stories and novels that you should definitely read if you get the chance)

# Why are you writing this blog?
There are two kinds of reasons why.

* *The altruistic reason*: 
    I'm very excited about machine learning, and as with most people who enjoy something, I also want to share that with everyone else. In particular, I think ML as a field is very diverse, with all kinds of subfields emerging everyday! 
    
    Navigating all of them can be challenging (but very rewarding), so I want to provide a unified resource where you can read about different fields, and come away with a better understanding of what's going on in each of them (or, for people new to ML/AI, a general understanding of how it works and affects them). 

* *The selfish reason*: As I've started to do more work in machine learning, I've realized that reading research papers, and communicating the key ideas from them is a very important skill to have:
    
    * It gives you the right tools to keep in touch with the latest happenings in ML (or any field for that matter). This is especially true in deep learning, a field that is fairly new and where most knowledge is not in textbooks but in the treasure trove of papers being published each year at the top conferences. 

    * The process of communicating those ideas to other people also makes you understand them even better than just reading them (sort of how teaching something grounds you better in the material).
    
    I believe that the process of creating and consistently updating a blog like this will definitely me build those skills, while also giving me (and my readers, hopefully) access to an armory of ideas we can all build upon in our research, projects and work. 

# Where can I share feedback on the blog? (What worked, what didn't, questions, etc)
I'm glad you asked! I've set up a form right below this, where you can enter your email, comment and "POST" a comment to me! 

<form action="https://formspree.io/kavishwarvyom@berkeley.edu" method="POST">
Your email<br><input type="email" name="_replyto"><br>

Comment<br><textarea name="body" rows="10" cols="50"></textarea><br><input type="submit" value="Send"></form>