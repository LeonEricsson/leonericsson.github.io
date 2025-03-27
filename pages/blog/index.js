import { getSortedPostsData } from "../../lib/posts";
import Link from "next/link";
import Layout from "../../components/Layout";
import { useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css"; // `rehype-katex` does not import the CSS for you

export async function getStaticProps() {
  const allPostsData = getSortedPostsData();
  return {
    props: {
      allPostsData,
    },
  };
}

export default function Blog({ allPostsData }) {
  const [currentPage, setCurrentPage] = useState(1);
  const postsPerPage = 5;
  const indexOfLastPost = currentPage * postsPerPage;
  const indexOfFirstPost = indexOfLastPost - postsPerPage;
  const currentPosts = allPostsData.slice(indexOfFirstPost, indexOfLastPost);

  const formatDate = (dateString) => {
    const options = { year: "numeric", month: "long", day: "numeric" };
    return new Date(dateString).toLocaleDateString("en-US", options);
  };

  const paginate = (pageNumber) => setCurrentPage(pageNumber);

  return (
    <Layout>
      <div className="container mx-auto py-12 px-4 sm:px-6 lg:px-8 flex justify-center">
        <div className="w-full lg:w-3/5">
          <p className={`text-base italic mb-4 font-serif`}>
          this is a space to document and distill my thoughts on publications
            and articles within the research community. literature reviews are personal
            and messy; blog post are intended for a wider audience (marked with a{" "}
              <span style={{ color: "#5688a9" }}>dot</span>).
            <br />
            <br />
            i write for my own edification; and posts are subject to my own interest
            for the topic at the time of writing.
          </p>
          <div className="vertical-line"></div>
          {currentPosts.map(({ id, date, title, excerpt, type }) => (
            <div key={id} className="pb-16">
              <Link href={`/blog/${id}`}>
                <span className="text-2xl cursor-pointer text-center block georgia font-bold text-headline">
                {type === "blog" && (
                    <span
                      style={{
                        height: "4px",
                        width: "4px",
                        backgroundColor: "#5688a9",
                        borderRadius: "50%",
                        display: "inline-block",
                        marginRight: "5px",
                        verticalAlign: "middle",
                      }}
                    ></span>
                  )}
                  {title}
                </span>
                <p className="text-gray-400 text-base text-center pb-4 pt-1">
                  {formatDate(date)}
                </p>
                <p className={`text-gray-600 text-l text-left pb-14 georgia`}>
                  <Markdown
                    remarkPlugins={[remarkGfm, remarkMath]}
                    rehypePlugins={[rehypeKatex]}
                    className="markdown georgia leading-loose "
                  >
                    {excerpt}
                  </Markdown>
                </p>
              </Link>
              <hr className="border-opacity-20 border-black" />
            </div>
          ))}
          <div className="pt-10 text-center">
            {[
              ...Array(Math.ceil(allPostsData.length / postsPerPage)).keys(),
            ].map((number) => (
              <button
                key={number}
                onClick={() => paginate(number + 1)}
                className="text-xl p-3 hover:bg-gray-200"
              >
                {number + 1}
              </button>
            ))}
          </div>
        </div>
      </div>
    </Layout>
  );
}
