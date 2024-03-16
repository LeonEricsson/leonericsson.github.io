import React from "react";
import Link from "next/link";
import Layout from "../components/Layout";
import { getSortedPostsData } from "../lib/posts";
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

export default function Index({ blogPosts, reviews }) { 
  return (
    <Layout>
    <div className="px-80 py-8">
      <h2 className="text-lg mt-8 mb-2 ml-24 font-semibold text-black inline-block px-1 rounded" style={{ backgroundColor: '#475770' }}>Blog posts</h2>
      <ul className="list-none">
        {blogPosts.map(({ id, title, date }) => (
          <li key={id} className="mt-2 flex items-center">
            <span className="text-gray-400 mr-4">{format(new Date(date), 'dd MMM yyyy')}</span>
            <Link href={`/blog/${id}`}>
              <span className=" hover:text-blue-800 cursor-pointer">{title}</span>
            </Link>
          </li>
        ))}
      </ul>

      <h2 className="text-lg mt-12 mb-2 ml-24 font-semibold text-black inline-block px-1 rounded" style={{ backgroundColor: '#fccccc' }}>Lit. reviews</h2>
      <ul className="list-none">
        {reviews.map(({ id, title, date }) => (
          <li key={id} className="mt-2 flex items-center">
            <span className="text-gray-400 mr-4">{format(new Date(date), 'dd MMM yyyy')}</span>
            <Link href={`/reviews/${id}`}>
              <span className=" hover:text-blue-800 cursor-pointer">{title}</span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
    </Layout>
  );
}

