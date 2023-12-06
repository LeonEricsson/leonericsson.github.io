import React from "react";
import Link from "next/link";

function AboutCard() {
  return (
    <div>
      <p>
        my name is leon ericsson. i'm a recent grad with a master's degree in machine
        learning living in stockholm with my <i><a href="https://collectum.se/en/startpage/private/your-situation/i-have-a-sambo#:~:text=%E2%80%9CSambo%E2%80%9D%20is%20a%20Swedish%20legal,cover%20and%2For%20Repayment%20cover.">sambo</a></i>.
        when i'm not working i enjoy unique cliches; cooking, the gym and
        traveling.
        </p>
        <br></br>
        <p>
        professionally, i'm a research engineer. my interests are scattered,
        feels like i find something new every other week, but broadly i'd say
        they fall into {" "} <b>foundational models</b>, <b>policy learning</b>,{" "}
        <b>computer vision</b>, and <b>medical ai</b>. if you're curious about
        my current research interests, skim my blog. i hacked this website as a platform to share
        and document my thoughts on research that i come across. while primarily for my own edification,
        hopefully there's something here that proves insightful to you.
      </p>
    </div>
  );
}

export default AboutCard;
