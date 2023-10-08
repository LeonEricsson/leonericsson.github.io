import React, { useEffect, useState } from "react";
import Typewriter from "typewriter-effect";

function Type({ onComplete }) {
  return (
    <Typewriter
      onInit={(typewriter) => {
        typewriter
          .typeString("hi")
          .pauseFor(1e3)
          .deleteAll()
          .typeString("I'm leon.")
          .pauseFor(4e3)
          .deleteAll()
          .typeString("the latest from my blog")
          .callFunction(() => {
            onComplete();
          })
          .start();
      }}
    />
  );
}

export default Type;
