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
          content="Leon Ericsson | Learning..."
        />
        <meta
          property="og:description"
          content="Personal website of Leon Ericsson"
        />
        <meta
          property="og:site_name"
          content="Leon Ericsson | Learning..."
        />
        <link rel="icon" href="/favicon.ico" />
        <link rel="manifest" href="/manifest.json" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Cardo:ital,wght@0,400;0,700;1,400&family=Crimson+Text:ital,wght@0,400;0,600;0,700;1,400&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Lora:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet" />
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
