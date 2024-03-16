import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import {remark} from 'remark';
import html from 'remark-html';

const postsDirectory = path.join(process.cwd(), 'blog');

export function getAllPostIds() {
  const fileNames = fs.readdirSync(postsDirectory);

  return fileNames.map(fileName => {
    return {
      params: {
        id: fileName.replace(/\.md$/, ''),
      },
    };
  });
}

export async function getPostData(id) {
  const fullPath = path.join(postsDirectory, `${id}.md`);
  const fileContents = fs.readFileSync(fullPath, 'utf8');
  
  const matterResult = matter(fileContents);
  const contentMarkdown = matterResult.content;

  const [year, month, day, ...rest] = id.split("-");
  const date = new Date(`${year}-${month}-${day}`);
  return {
    id,
    contentMarkdown,
    date: date.toISOString(),
    ...matterResult.data,
  };
}

export function getSortedPostsData() {
  const fileNames = fs.readdirSync(postsDirectory);
  const allPostsData = fileNames.map(fileName => {
    const [year, month, day, ...rest] = fileName.split("-");
    const name = fileName.replace(/\.md$/, '')

    const date = new Date(`${year}-${month}-${day}`);

    const fullPath = path.join(postsDirectory, fileName);
    const fileContents = fs.readFileSync(fullPath, "utf8");
    const matterResult = matter(fileContents);
    const words = matterResult.content.split(/\s+/).slice(0, 90).join(" ");

    return {
      id: name,
      excerpt: words + "...",
      date: date.toISOString(),
      ...matterResult.data,
    };
  });
  return allPostsData.sort((a, b) => new Date(b.date) - new Date(a.date));
}

