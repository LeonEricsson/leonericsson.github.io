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
  <div className="relative pl-[25rem] py-12">
    {/* Top-right narrow paragraph */}
    <div className="absolute top-12 right-8 w-[18rem]">
      <p className="text-sm font-lora text-medium-gray leading-relaxed">
        this is a space to document and distill my thoughts on publications
        and articles within the research community. literature reviews are personal
        and messy; blog posts are slightly more refined.
        <br />
        <br />
        i write for my own edification - posts are subject to my own interests
      </p>
    </div>

    {/* Main blog section */}
    <h2 className="text-4xl mt-8 mb-6 font-sorts-mill italic text-deep-charcoal">
      Blog posts
    </h2>
    <div className="w-16 h-px bg-subtle-line mb-8"></div>
    <ul className="list-none space-y-4">
      {blogPosts.map(({ id, title, date }) => (
        <li key={id} className="flex items-baseline">
          <span className="text-medium-gray text-sm font-crimson mr-6 min-w-[6rem]">{format(new Date(date), 'dd MMM yyyy')}</span>
          <Link href={`/blog/${id}`}>
            <span className="text-deep-charcoal font-sorts-mill italic text-lg hover:text-sepia-accent cursor-pointer transition-colors duration-300">{title}</span>
          </Link>
        </li>
      ))}
    </ul>

    <h2 className="text-4xl mt-16 mb-6 font-sorts-mill italic text-deep-charcoal">
      Literature reviews
    </h2>
    <div className="w-16 h-px bg-subtle-line mb-8"></div>
    <ul className="list-none space-y-4">
      {reviews.map(({ id, title, date }) => (
        <li key={id} className="flex items-baseline">
          <span className="text-medium-gray text-sm font-crimson mr-6 min-w-[6rem]">{format(new Date(date), 'dd MMM yyyy')}</span>
          <Link href={`/blog/${id}`}>
            <span className="text-deep-charcoal font-sorts-mill italic text-lg hover:text-sepia-accent cursor-pointer transition-colors duration-300">{title}</span>
          </Link>
        </li>
      ))}
    </ul>
  </div>
</Layout>

  );
}

