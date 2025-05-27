import React from "react";
import Link from "next/link";
import Layout from "../../components/Layout";
import { getSortedPostsData } from "../../lib/posts";
import { format } from 'date-fns'; 

export async function getStaticProps() {
  const allPostsData = getSortedPostsData();

  const blogPosts = allPostsData.filter((post) => post.type === "blog");
  const reviews = allPostsData.filter((post) => post.type === "paper");

  return {
    props: {
      blogPosts,
      reviews,
    },
  };
}

export default function Blog({ blogPosts, reviews }) { 
  return (
    <Layout>
  <div className="relative pl-[25rem] py-4">
    {/* Top-right narrow paragraph */}
    <div className="absolute top-8 right-4 w-[18rem]">
      <p className="text-xs italic palatino">
        this is a space to document and distill my thoughts on publications
        and articles within the research community. literature reviews are personal
        and messy; blog posts are slightly more refined.
        <br />
        <br />
        i write for my own edification - posts are subject to my own interests
      </p>
    </div>

    {/* Main blog section */}
    <h2 className="text-xl mt-8 mb-2 ml-18 font-bold text-headline inline-block px-1 rounded">
      Blog posts
    </h2>
    <ul className="list-none">
      {blogPosts.map(({ id, title, date }) => (
        <li key={id} className="mt-2 flex items-center">
          <span className="text-gray-400 mr-4">{format(new Date(date), 'dd MMM yyyy')}</span>
          <Link href={`/blog/${id}`}>
            <span className="hover:text-blue-800 text-gray-800 cursor-pointer">{title}</span>
          </Link>
        </li>
      ))}
    </ul>

    <h2 className="text-xl mt-12 mb-1 ml-18 font-bold text-headline inline-block px-1 rounded">
      Literature reviews
    </h2>
    <ul className="list-none">
      {reviews.map(({ id, title, date }) => (
        <li key={id} className="mt-2 flex items-center">
          <span className="text-gray-400 mr-4">{format(new Date(date), 'dd MMM yyyy')}</span>
          <Link href={`/blog/${id}`}>
            <span className="hover:text-blue-800 text-gray-800 cursor-pointer">{title}</span>
          </Link>
        </li>
      ))}
    </ul>
  </div>
</Layout>

  );
}

