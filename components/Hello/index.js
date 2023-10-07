import React, { useState, useEffect } from "react";
import Type from "../Type"; 

export default function Hello({allPostsData}) {
  const [showBlogs, setShowBlogs] = useState(false);

  // Placeholder for blog data
  const blogs = allPostsData.slice(0,5)

  const handleTypingComplete = () => {
    setShowBlogs(true);
  };

  // Fixed positions for the blog cards
  const fixedPositions = [
    { x: -190, y: -110 },
    { x: 120, y: 330 },
    { x: 520, y: -120 },
    { x: 580, y: 220 },
    { x: -260, y: 260 },

  ];

  useEffect(() => {
    if (showBlogs) {
      const baseDelay = 1000; // 1 second base delay for all cards
      const elements = document.querySelectorAll('.blog-card');
      elements.forEach((el, index) => {
        setTimeout(() => {
          el.style.opacity = '1';
        }, baseDelay + (index * 1000)); // 1 second increment for each subsequent card
      });
    }
  }, [showBlogs]);

  return (
    <div className="relative pb-5">
      <h1 className="text-large pt-4 md:pt-12 lg:pt-24 pl-12 md:text-4xl text-white">
        <Type onComplete={handleTypingComplete} />
      </h1>
      {showBlogs && (
        <div className="absolute top-0 left-0">
          {blogs.map((blog, index) => {
            const { x, y } = fixedPositions[index];
            return (
              <a
                href={"blog/" + blog.id}
                key={index}
                className="blog-card block absolute transition-opacity duration-700 ease-in opacity-0 min-w-[250px]"
                style={{ top: `${y}px`, left: `${x}px` }}
              >
                <div className="p-4 rounded bg-opacity-50 bg-warm-cream">
                  <h2 className="font-bold text-base truncate-3-lines hover:text-gray-600" title={blog.title}>{blog.title}</h2>
                </div>
              </a>
            );
          })}
        </div>
      )}
    </div>
  );
}

