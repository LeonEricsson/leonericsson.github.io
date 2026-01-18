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
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark-dimmed.css";

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

      <article className="container mx-auto py-16 px-6 sm:px-8 lg:px-12 flex justify-center">
        <div className="w-full max-w-3xl">
          {/* Decorative top line */}
          <div className="w-16 h-px bg-subtle-line mx-auto mb-12"></div>

          <h1 className="text-6xl font-cormorant font-medium text-deep-charcoal text-center mb-6 leading-tight">
            {postData.title}
          </h1>

          <p className="text-medium-gray text-sm text-center font-crimson pb-12 pt-2">
            {postData.author ? (
              <>
                Original{" "}
                <a
                  href={postData.exturl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sepia-accent hover:underline transition-all duration-300"
                >
                  paper
                </a>{" "}
                · {postData.author} et al {postData.year}
              </>
            ) : (
              <>{formatDate(postData.date)}</>
            )}
          </p>

          {/* Decorative separator line */}
          <div className="w-24 h-px bg-subtle-line mx-auto mb-16"></div>

          <div className="entry text-base text-deep-charcoal font-lora leading-loose">
            <Markdown
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[
                rehypeKatex,
                rehypeRaw,
                [rehypeHighlight, { detect: true }]
              ]}
              className="markdown"
            >
              {postData.contentMarkdown}
            </Markdown>
          </div>

          {/* Decorative bottom line */}
          <div className="w-16 h-px bg-subtle-line mx-auto mt-20"></div>
        </div>
      </article>
    </LayoutWoNav>
  );
}
