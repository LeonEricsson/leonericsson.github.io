import { getAllPostIds, getPostData } from "../../lib/posts";
import Layout from "../../components/Layout";
import LayoutWoNav from "../../components/LayoutWoNav";
import Head from "next/head";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import "katex/dist/katex.min.css"; // `rehype-katex` does not import the CSS for you

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
    <LayoutWoNav>
      <Head></Head>

      <article className="container mx-auto py-20 px-4 sm:px-6 lg:px-8 flex justify-center">
        <div className="w-full lg:w-4/5">
          {" "}
          {/* Same proportions as in Blog component */}
          <h1 className="text-4xl font-bold mb-4 charter text-center">
            {postData.title}
          </h1>
          <p className="text-gray-400 text-base text-center palatino pb-20 pt-1">
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
          <div className={`entry text-base text-gray-600 mb-4 palatino leading-relaxed`}>
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
    </LayoutWoNav>
  );
}
