import React from "react";
import Typewriter from "typewriter-effect";

function Type() {
  return (
    <div className="flex items-center">
      <Typewriter
        onInit={(typewriter) => {
          typewriter
            .typeString("hi")
            .pauseFor(1500)
            .deleteAll()
            .typeString("i'm leon.")
            .start();
        }}
      />
    </div>
  );
}

export default Type;