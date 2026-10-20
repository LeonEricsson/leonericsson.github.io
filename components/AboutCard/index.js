import React from "react";

function AboutCard() {
  return (
    <div>
      <h2 className="text-3xl font-sorts-mill italic text-deep-charcoal pb-6">about me</h2>
      <p className="font-lora leading-relaxed text-deep-charcoal">
        hi! i'm leon, i'm a research engineer living in stockholm. i've got a background in software engineering, with a master's in compsci but now adays i work in research related to reinforcement learning and language models. overall i'd say my interests are well reflected in my blog/notes.
      </p>
      <br />
      <p className="font-lora leading-relaxed text-deep-charcoal">
        otherwise i love everything food related, and traveling to new places.
      </p>
      <h2 className="text-3xl font-sorts-mill italic text-deep-charcoal pb-6 pt-16">i like open source</h2>
        <p className="font-lora leading-relaxed text-deep-charcoal">
          selection of my favorites, some of which i've worked on.
          <ul className="list-none ml-4 pt-2">
            <li className="before:content-['-'] before:pr-2">rl with <a href="https://github.com/PrimeIntellect-ai/prime-rl" className="text-sepia-accent hover:underline transition-all duration-300">prime-rl</a>, or <a href="https://github.com/THUDM/slime" className="text-sepia-accent hover:underline transition-all duration-300">slime</a> </li>
            <li className="before:content-['-'] before:pr-2"><a href="https://github.com/huggingface/trl" className="text-sepia-accent hover:underline transition-all duration-300">trl</a> has a special place in my heart</li>
            <li className="before:content-['-'] before:pr-2"><a href="https://github.com/vllm-project/vllm" className="text-sepia-accent hover:underline transition-all duration-300">vllm</a> </li>
            <li className="before:content-['-'] before:pr-2">ggerganov's <a href="https://github.com/ggerganov/llama.cpp" className="text-sepia-accent hover:underline transition-all duration-300">llama.cpp</a></li>
            <li className="before:content-['-'] before:pr-2"><a href="https://github.com/google/jax" className="text-sepia-accent hover:underline transition-all duration-300">jax</a>: lovechild of autograd and xla</li>
            <li className="before:content-['-'] before:pr-2"><a href="https://github.com/tmux/tmux" className="text-sepia-accent hover:underline transition-all duration-300">tmux</a></li>
          </ul>
        </p>
      <h2 className="text-3xl font-sorts-mill italic text-deep-charcoal pb-6 pt-16">i like making things</h2>
        <p className="font-lora leading-relaxed text-deep-charcoal">
        mixed bag, somewhat outdated
        <ul className="list-none ml-4 pt-2">
          <li className="before:content-['-'] before:pr-2">paper implementations in <a href="https://github.com/LeonEricsson/omni" className="text-sepia-accent hover:underline transition-all duration-300">omni</a></li>
          <li className="before:content-['-'] before:pr-2">this website, including my <a href="https://leonericsson.github.io/blog" className="text-sepia-accent hover:underline transition-all duration-300">blog</a></li>
          <li className="before:content-['-'] before:pr-2"><a href="https://github.com/LeonEricsson/mindex" className="text-sepia-accent hover:underline transition-all duration-300">mindex</a>, a local semantic search engine for your mind index</li>
          <li className="before:content-['-'] before:pr-2"><a href="https://github.com/LeonEricsson/alphax" className="text-sepia-accent hover:underline transition-all duration-300">alphax</a>, scalable + fast alphazero in jax. facilitates spmd, simd, jit-composition etc </li>
          <li className="before:content-['-'] before:pr-2"><a href="https://github.com/LeonEricsson/llmjudge" className="text-sepia-accent hover:underline transition-all duration-300">exploring</a> limitations of llm-as-a-judge</li>
          <li className="before:content-['-'] before:pr-2"><a href="https://github.com/LeonEricsson/llmcontext" className="text-sepia-accent hover:underline transition-all duration-300">llmcontext</a> (fork of <a href="https://github.com/gkamradt/LLMTest_NeedleInAHaystack" className="text-sepia-accent hover:underline transition-all duration-300">original</a>), a context window pressure tester of <b>open</b> llms</li>
          <li className="before:content-['-'] before:pr-2">a <a href="https://github.com/LeonEricsson/Ensemble4DFlowNet" className="text-sepia-accent hover:underline transition-all duration-300">framework</a> for super-resolution of clinical 4d flow mri. <a href="https://arxiv.org/abs/2311.11819" className="text-sepia-accent hover:underline transition-all duration-300">paper</a>.</li>

        </ul>
      </p>

        <h2 className="text-3xl font-sorts-mill italic text-deep-charcoal pb-6 pt-16">i like reading</h2>
        <p className="font-lora leading-relaxed text-deep-charcoal">
          some favorites of mine
          <ul className="list-none ml-4 pt-2">
            <li className="before:content-['-'] before:pr-2"><a href="https://physics.allen-zhu.com/" className="text-sepia-accent hover:underline transition-all duration-300">Physics of Language Models</a> </li>
            <li className="before:content-['-'] before:pr-2"><a href="https://arxiv.org/abs/2505.24298" className="text-sepia-accent hover:underline transition-all duration-300">AReaL</a> </li>
            <li className="before:content-['-'] before:pr-2"><a href="https://arxiv.org/pdf/2402.10588" className="text-sepia-accent hover:underline transition-all duration-300">On the Latent Language of Multilingual Transformers</a> </li>
            <li className="before:content-['-'] before:pr-2"><a href="https://arxiv.org/abs/1911.08265" className="text-sepia-accent hover:underline transition-all duration-300">MuZero</a> </li>
            <li className="before:content-['-'] before:pr-2"><a href="https://arxiv.org/pdf/2405.04434" className="text-sepia-accent hover:underline transition-all duration-300">DeepSeek-V2</a> </li>
            <li className="before:content-['-'] before:pr-2"><a href="https://arxiv.org/abs/2508.06471" className="text-sepia-accent hover:underline transition-all duration-300">GLM 4.5</a> </li>
            <li className="before:content-['-'] before:pr-2"><a href="https://jax-ml.github.io/scaling-book/" className="text-sepia-accent hover:underline transition-all duration-300">How to Scale Your Model</a> </li>
            <li className="before:content-['-'] before:pr-2"><a href="https://spaces.ac.cn/archives/10091" className="text-sepia-accent hover:underline transition-all duration-300">MHA, MQA, GQA, MLA (Jianlin Su)</a> </li>
            <li className="before:content-['-'] before:pr-2"><a href="https://transformer-circuits.pub/" className="text-sepia-accent hover:underline transition-all duration-300">Transformer Circuits</a> </li>
          </ul>
        </p>

        <h2 className="text-3xl font-sorts-mill italic text-deep-charcoal pb-6 pt-16">ideas</h2>
          <p className="font-lora leading-relaxed text-deep-charcoal">
            things / projects i'm thinking about lately.
            <ul className="list-none ml-4 pt-2">
              <li className="before:content-['-'] before:pr-2">visual reasoning and visual spatial reasoning. non-text cot?</li>
              <li className="before:content-['-'] before:pr-2">proper long-context reasoning benchmark. forcing retrieval + reasoning across the context</li>
              <li className="before:content-['-'] before:pr-2">async RL framework optimized for single node training</li>
            </ul>
          </p>


    </div>
  );
}

export default AboutCard;
