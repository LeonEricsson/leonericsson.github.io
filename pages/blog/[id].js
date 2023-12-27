import { getAllPostIds, getPostData } from "../../lib/posts";
import Layout from "../../components/Layout";
import Head from "next/head";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import "katex/dist/katex.min.css"; // `rehype-katex` does not import the CSS for you
import Image from "next/image";

export async function getStaticPaths() {
  const paths = getAllPostIds();
  return {
    paths,
    fallback: false,
  };
}

export async function getStaticProps({ params }) {
  const postData = await getPostData(params.id);
  return {
    props: {
      postData,
    },
  };
}

const formatDate = (dateString) => {
  const options = { year: "numeric", month: "long", day: "numeric" };
  return new Date(dateString).toLocaleDateString("en-US", options);
};

export default function Post({ postData }) {
  return (
    <Layout>
      <Head></Head>

      <article className="container mx-auto py-12 px-4 sm:px-6 lg:px-8 flex justify-center">
        <div className="w-full lg:w-3/5">
          {" "}
          {/* Same proportions as in Blog component */}
          <h1 className="text-4xl font-bold mb-4 text-center">
            {postData.title}
          </h1>
          <p className="text-gray-400 text-base text-center pb-4 pt-1">
            {postData.author ? (
              <>
                Original{" "}
                <a
                  href={postData.exturl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-400 hover:text-blue-300"
                >
                  paper
                </a>{" "}
                · {postData.author} et al {postData.year}
              </>
            ) : (
              <>{formatDate(postData.date)}.</>
            )}
          </p>
          <div className="entry text-lg mb-4">
            <Markdown
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeKatex, rehypeRaw]}
              className="markdown"
            >
              {postData.contentMarkdown}
            </Markdown>
          </div>
        </div>
      </article>
    </Layout>
  );
}
