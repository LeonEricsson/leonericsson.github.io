import React from "react";

function AboutCard() {
  return (
    <div>
      <h2 className="text-3xl font-sorts-mill italic text-deep-charcoal pb-6">about me</h2>
      <p className="font-lora leading-relaxed text-deep-charcoal">
        hi! i'm leon, i'm a research engineer living in stockholm, lived here for almost 2 years now with my fiance(!). big fan of ml research, and just understanding how things work. otherwise i enjoy eating food, making food, traveling and working out.
      </p>
      <br />
      <p className="font-lora leading-relaxed text-deep-charcoal">
        i've got a background in software engineering, with a master's in compsci. but today i'm focused on <b>language models</b>, <b>reinforcement learning</b>, <b>mech interp</b>, and <b>reasoning</b>. although my experience is limited i'm very keen on health/medicine as a field of application for the above. if you're curious about my current fixations, skim the blog.
      </p>
      <h2 className="text-3xl font-sorts-mill italic text-deep-charcoal pb-6 pt-16">i like open source</h2>
        <p className="font-lora leading-relaxed text-deep-charcoal">
          these are my favorite, some of which i contribute to.
          <ul className="list-none ml-4 pt-2">
            <li className="before:content-['-'] before:pr-2">rl with <a href="https://github.com/huggingface/trl" className="text-sepia-accent hover:underline transition-all duration-300">trl</a> (<a href="https://github.com/huggingface/trl/" className="text-sepia-accent hover:underline transition-all duration-300">core maintainer</a>) </li>
            <li className="before:content-['-'] before:pr-2">inference engine <a href="https://github.com/vllm-project/vllm" className="text-sepia-accent hover:underline transition-all duration-300">vllm</a> </li>
            <li className="before:content-['-'] before:pr-2">finetuning with <a href="https://github.com/unslothai/unsloth" className="text-sepia-accent hover:underline transition-all duration-300">unsloth</a> </li>
            <li className="before:content-['-'] before:pr-2">ggerganov's <a href="https://github.com/ggerganov/llama.cpp" className="text-sepia-accent hover:underline transition-all duration-300">llama.cpp</a> (<a href="https://github.com/ggerganov/llama.cpp/pull/4484" className="text-sepia-accent hover:underline transition-all duration-300">#4484</a>)</li>
            <li className="before:content-['-'] before:pr-2"><a href="https://github.com/ml-explore/mlx" className="text-sepia-accent hover:underline transition-all duration-300">mlx</a> is an array framework for apple silicon (<a href="https://github.com/ml-explore/mlx-examples/pull/202" className="text-sepia-accent hover:underline transition-all duration-300">#202</a> <a href="https://github.com/ml-explore/mlx-examples/pull/237" className="text-sepia-accent hover:underline transition-all duration-300">#237</a> <a href="https://github.com/ml-explore/mlx/pull/456" className="text-sepia-accent hover:underline transition-all duration-300">#456</a> <a href="https://github.com/ml-explore/mlx-examples/pull/19" className="text-sepia-accent hover:underline transition-all duration-300">#19</a> <a href="https://github.com/ml-explore/mlx-examples/pull/276" className="text-sepia-accent hover:underline transition-all duration-300">#276</a>) </li>
            <li className="before:content-['-'] before:pr-2"><a href="https://github.com/LeonEricsson/Tensor-Puzzles" className="text-sepia-accent hover:underline transition-all duration-300">tensor puzzles</a> and <a href="https://github.com/LeonEricsson/GPU-Puzzles" className="text-sepia-accent hover:underline transition-all duration-300">cuda puzzles</a> from sasha rush</li>
            <li className="before:content-['-'] before:pr-2">community driven <a href="https://github.com/ohmyzsh/ohmyzsh" className="text-sepia-accent hover:underline transition-all duration-300">ohmyzsh</a></li>
            <li className="before:content-['-'] before:pr-2"><a href="https://github.com/google/jax" className="text-sepia-accent hover:underline transition-all duration-300">jax</a>: the lovechild of autograd and xla</li>
          </ul>
        </p>
      <h2 className="text-3xl font-sorts-mill italic text-deep-charcoal pb-6 pt-16">i like making things</h2>
        <p className="font-lora leading-relaxed text-deep-charcoal">
        healthy mix of research, experiments, and scratch projects.
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

        <h2 className="text-3xl font-sorts-mill italic text-deep-charcoal pb-6 pt-16">i like reading papers</h2>
        <p className="font-lora leading-relaxed text-deep-charcoal">
          some favorites of mine.
          <ul className="list-none ml-4 pt-2">
            <li className="before:content-['-'] before:pr-2"><a href="https://physics.allen-zhu.com/" className="text-sepia-accent hover:underline transition-all duration-300">Physics of Language Models</a> </li>
            <li className="before:content-['-'] before:pr-2"><a href="https://arxiv.org/pdf/2402.10588" className="text-sepia-accent hover:underline transition-all duration-300">On the Latent Language of Multilingual Transformers</a> </li>
            <li className="before:content-['-'] before:pr-2"><a href="https://transformer-circuits.pub/" className="text-sepia-accent hover:underline transition-all duration-300">Transformer Circuits</a> </li>
            <li className="before:content-['-'] before:pr-2"><a href="https://arxiv.org/abs/2508.06471" className="text-sepia-accent hover:underline transition-all duration-300">GLM 4.5</a> </li>
            <li className="before:content-['-'] before:pr-2"><a href="https://arxiv.org/abs/2505.24298" className="text-sepia-accent hover:underline transition-all duration-300">AReaL</a> </li>
            <li className="before:content-['-'] before:pr-2"><a href="https://arxiv.org/abs/2508.08221" className="text-sepia-accent hover:underline transition-all duration-300">Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning </a> </li>
            <li className="before:content-['-'] before:pr-2"><a href="https://cohere.com/research/papers/command-a-technical-report.pdf" className="text-sepia-accent hover:underline transition-all duration-300">Command-A</a> </li>
            <li className="before:content-['-'] before:pr-2"><a href="https://arxiv.org/pdf/2405.04434" className="text-sepia-accent hover:underline transition-all duration-300">DeepSeek-V2</a> </li>
            <li className="before:content-['-'] before:pr-2"><a href="https://arxiv.org/abs/1911.08265" className="text-sepia-accent hover:underline transition-all duration-300">MuZero</a> </li>
            <li className="before:content-['-'] before:pr-2"><a href="https://spaces.ac.cn/archives/10091" className="text-sepia-accent hover:underline transition-all duration-300">MHA, MQA, GQA, MLA (by Jianlin Su)</a> </li>
            <li className="before:content-['-'] before:pr-2"><a href="https://arxiv.org/pdf/2306.00978" className="text-sepia-accent hover:underline transition-all duration-300">Activation-aware weight quantization</a> </li>
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
