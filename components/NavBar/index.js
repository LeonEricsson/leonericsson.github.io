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
        className={`inline-flex items-center px-4 pt-8 pb-3 text-base font-sorts-mill italic text-deep-charcoal transition-all duration-300 ${
          isActive ? 'border-b-2 border-deep-charcoal' : 'hover:text-sepia-accent'
        }`}
      >
        {children}
      </Link>
    );
  };
  
  return (
    <Disclosure as="nav" className="bg-warm-cream border-b border-subtle-line">
      {({ open }) => (
        <>
          <div className="mx-auto max-w-7xl px-2 sm:px-4 lg:px-8">
            <div className="flex h-20 justify-center items-center">
              <div className="hidden items-center md:flex md:space-x-12">
                {/* only show this tabs when size is > md */}
                <NavLink href="/">home</NavLink>
                <NavLink href="/about">about</NavLink>
                <NavLink href="/blog">blog</NavLink>
                <NavLink href="/library">library</NavLink>
                <a
                  href="https://github.com/LeonEricsson?tab=repositories"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center px-4 pt-8 pb-3 text-base font-sorts-mill italic text-deep-charcoal hover:text-sepia-accent transition-all duration-300"
                >
                  work
                </a>
              </div>
              <div className="flex items-center md:hidden">
                {/* Mobile menu button */}
                <Disclosure.Button className="inline-flex items-center justify-center rounded-md p-2 text-deep-charcoal hover:bg-vintage-paper focus:outline-none focus:ring-2 focus:ring-inset focus:ring-sepia-accent">
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

          <Disclosure.Panel className="md:hidden bg-warm-cream">
            <div className="space-y-1 pt-2 pb-3">
              <Disclosure.Button
                as={Link}
                href="/"
                className="block border-l-2 border-subtle-line py-3 pl-4 pr-4 text-sm font-sorts-mill italic text-deep-charcoal hover:bg-vintage-paper hover:border-sepia-accent transition-all duration-300"
              >
                Home
              </Disclosure.Button>
              <Disclosure.Button
                as={Link}
                href="/blog"
                className="block border-l-2 border-subtle-line py-3 pl-4 pr-4 text-sm font-sorts-mill italic text-deep-charcoal hover:bg-vintage-paper hover:border-sepia-accent transition-all duration-300"
              >
                Blog
              </Disclosure.Button>
              <Disclosure.Button
                as={Link}
                href="/about"
                className="block border-l-2 border-subtle-line py-3 pl-4 pr-4 text-sm font-sorts-mill italic text-deep-charcoal hover:bg-vintage-paper hover:border-sepia-accent transition-all duration-300"
              >
                About
              </Disclosure.Button>
              <Disclosure.Button
                as={Link}
                href="/library"
                className="block border-l-2 border-subtle-line py-3 pl-4 pr-4 text-sm font-sorts-mill italic text-deep-charcoal hover:bg-vintage-paper hover:border-sepia-accent transition-all duration-300"
              >
                Library
              </Disclosure.Button>
            </div>
          </Disclosure.Panel>
        </>
      )}
    </Disclosure>
  );
}
