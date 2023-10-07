import { Html, Head, Main, NextScript } from "next/document";

export default function Document() {
  return (
    <Html>
      <Head>
        <meta charSet="utf-8" />
        <meta
          name="description"
          content="Personal website of Leon Ericsson"
        />
        <meta
          name="keywords"
          content="Leon Ericsson, AI, ML, Research, NLP, RL"
        ></meta>
        <meta name="theme-color" content="#000000" />
        <meta property="og:type" content="website" />
        <meta
          property="og:title"
          content="Leon Ericsson | Deep learner"
        />
        <meta
          property="og:description"
          content="Personal website of Leon Ericsson"
        />
        <meta
          property="og:site_name"
          content="Leon Ericsson | Deep learner"
        />
        <link rel="icon" href="/favicon.ico" />
        <link rel="manifest" href="/manifest.json" />
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
