import React, { useEffect, useState } from "react";
import Typewriter from "typewriter-effect";

function Type({ onComplete }) {
  return (
    <Typewriter
      onInit={(typewriter) => {
        typewriter
          .typeString("hi")
          .pauseFor(1.5e3)
          .deleteAll()
          .typeString("i'm leon.")
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
