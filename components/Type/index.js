import React, { useEffect, useState } from "react";
import Typewriter from "typewriter-effect";

function Type({}) {
  return (
    <Typewriter
      onInit={(typewriter) => {
        typewriter
          .typeString("hi")
          .pauseFor(1.5e3)
          .deleteAll()
          .typeString("i'm leon.")
          .start();
      }}
    />
  );
}

export default Type;
