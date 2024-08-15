import { Disclosure } from "@headlessui/react";
import { Bars3Icon, XMarkIcon } from "@heroicons/react/24/outline";
import Link from "next/link";
import { useRouter } from 'next/router';

export default function NavBar() {

  const router = useRouter();

  const NavLink = ({ href, children }) => {
    const isActive = router.pathname === href;
    return (
      <Link
        href={href}
        className={`inline-flex items-center px-1 pt-8 pb-3 text-sm text-black transition-all duration-200 ${
          isActive ? 'font-bold' : 'font-medium hover:font-bold'
        }`}
      >
        {children}
      </Link>
    );
  };
  
  return (
    <Disclosure as="nav" className="bg-warm-cream shadow">
      {({ open }) => (
        <>
          <div className="mx-auto max-w-7xl px-2 sm:px-4 lg:px-8">
            <div className="flex h-20 justify-center">
              <div className="hidden items-center md:flex md:space-x-8">
                {/* only show this tabs when size is > md */}
                <NavLink href="/">home</NavLink>
                <NavLink href="/about">about</NavLink>
                <NavLink href="/blog">blog</NavLink>
                <NavLink href="/indexer">index</NavLink>
                <NavLink href="/publications">publications</NavLink>
                <a
                  href="https://github.com/LeonEricsson?tab=repositories"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center px-1 pt-8 pb-3 text-sm font-medium text-black hover:font-bold transition-all duration-200"
                >
                  projects
                </a>
              </div>
              <div className="flex items-center md:hidden">
                {/* Mobile menu button */}
                <Disclosure.Button className="inline-flex items-center justify-center rounded-md p-2 text-gray-400 hover:bg-gray-100 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-indigo-500">
                  <span className="sr-only">Open main menu</span>
                  {open ? (
                    <XMarkIcon className="block h-6 w-6" aria-hidden="true" />
                  ) : (
                    <Bars3Icon className="block h-6 w-6" aria-hidden="true" />
                  )}
                </Disclosure.Button>
              </div>
            </div>
          </div>

          <Disclosure.Panel className="md:hidden">
            <div className="space-y-1 pt-2 pb-3">
              {/* Current: "bg-indigo-50 border-indigo-500 text-indigo-700", Default: "border-transparent text-gray-600 hover:bg-gray-50 hover:border-gray-300 hover:text-gray-800" */}
              <Disclosure.Button
                as={Link}
                href="/"
                className="block border-l-4 border-b py-2 pl-3 pr-4 text-base font-medium text-white hover:bg-white hover:text-black"
              >
                Home
              </Disclosure.Button>
              <Disclosure.Button
                as={Link}
                href="/blog"
                className="block border-l-4 border-b py-2 pl-3 pr-4 text-base font-medium text-white hover:bg-white hover:text-black"
              >
                Blogs
              </Disclosure.Button>
              <Disclosure.Button
                as={Link}
                href="/about"
                className="block border-l-4 border-b py-2 pl-3 pr-4 text-base font-medium text-white hover:bg-white hover:text-black"
              >
                About
              </Disclosure.Button>
              <Disclosure.Button
                as={Link}
                href="/publications"
                className="block border-l-4 border-b py-2 pl-3 pr-4 text-base font-medium text-white hover:bg-white hover:text-black"
              >
                Publications
              </Disclosure.Button>
              <Disclosure.Button
                as={Link}
                href="https://github.com/FrankLeeeee?tab=repositories"
                target="_blank"
                className="block border-l-4 border-b py-2 pl-3 pr-4 text-base font-medium text-white hover:bg-white hover:text-black"
              >
                Projects
              </Disclosure.Button>
            </div>
          </Disclosure.Panel>
        </>
      )}
    </Disclosure>
  );
}
