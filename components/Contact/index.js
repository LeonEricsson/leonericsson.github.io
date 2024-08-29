import React from "react";
import { contacts } from "../../data";
import Link from "next/link";

function Contact() {
  return (
    <div id="introduction">
      <div className="flex flex-row justify-center">
        {contacts.map((item, idx) => (
          <div
            key={idx}
            className="flex items-center justify-center w-10 h-10 mx-3"
          >
            <Link href={item.url} target="_blank">
              <item.icon className="w-5 h-5 text-headline" />
            </Link>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Contact;