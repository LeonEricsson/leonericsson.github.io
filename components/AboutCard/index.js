import React from "react";
import Link from "next/link";

function AboutCard() {
  return (
    <div>
      <p>
        My name is Leon Ericsson. I recently earned a Master's Degree in Machine
        Learning and have relocated to Stockholm with my <i>sambo</i>—a Swedish
        term that roughly translates to 'cohabitant' (ugh) in English. When I'm
        not researching I enjoy common cliches such as cooking, working out and
        traveling.
      </p>
      <br></br>
      <p>
        In the professional realm, I work as a Research Engineer. My interests
        span a wide range of topics, but I mostly focus on{" "}
        <b>foundational models</b>, <b>policy learning</b>,{" "}
        <b>computer vision</b>, and <b>medical AI</b>. If you're curious about
        my current research interests, the best way to find out is to explore my
        blog. I hacked this website over a weekend as a platform to share
        summaries and thoughts on the latest (sometimes seminal) research my
        areas of focus. While primarily for my own edification, I hope it offers
        valuable insights to anyone interested in these rapidly evolving fields.
      </p>
    </div>
  );
}

export default AboutCard;
