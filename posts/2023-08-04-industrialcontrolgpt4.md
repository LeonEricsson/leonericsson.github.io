---
layout: post
title: "Pre-Trained Large Language Models for Industrial Control"
categories: [Transformers]
year: 2023
type: paper
author: Song
exturl: https://arxiv.org/pdf/2308.03028.pdf
---
Good morning, afternoon or evening! I've got a short write-up today as I'm pressed for time. We're taking a look at an exciting real-world application of foundational models, unlike anything I've covered before on this blog I think. Authored by some brilliant minds at Microsoft Research Asia, this paper explores how pre-trained foundation models can revolutionize industrial control challenges. Imagine, GPT-4 controlling an entire Heating, Ventilation and Air Conditioning (HVAC) system without any human intervention. Insane that this is where we're at now...

The introduction of the paper presents a compelling argument for the need to rethink traditional methods of industrial control, particularly in scenarios where high-performance controllers are required with minimal samples and technical debt. We've all heard about the sample inefficiency issues plaguing reinforcement learning (RL), making it a costly affair in terms of both time and resources.

What struck me here was the analogy with humans becoming experts in a domain, which sometimes takes thousands of hours of experience. The authors put forth the idea that leveraging prior knowledge from foundation models could potentially bridge the gap between traditional control methods and the fast-paced demands of industrial control tasks. What's really exciting is the approach they're focused on: using pre-trained models directly. They suggest that these models, like GPT-4, are already packed with knowledge that can be harnessed for control tasks. It's like having a super-smart assistant ready to tackle complex problems! I believe the method is best described by the following figure, taken directly from the paper. It's a great overview of the entire system pipeline and gives intuition into how these models can be exploited by clever prompt engineering.

![](/images/industrialgpt4.png)

In a nutshell, this paper showcases a whole new way of looking at industrial control. It's about tapping into pre-existing AI wisdom to solve problems faster and smarter. The potential is enormous, and this paper is just the beginning. As I wrapped up reading, I couldn't help but be excited about the possibilities that lie ahead. AI-powered control systems might soon become as common as the devices they manage. Let's keep our eyes on this space, because big things are coming!