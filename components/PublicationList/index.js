import { publications } from "../../data";

function Techstack() {
  return (
    <div>
      {publications.map((publication, idx) => {
        return (
          <div className="text-black mb-1" key={idx}>
            <div className="mt-1 ml-8 my-2 flex flex-col">
              {publication.items.map((val, idx) => (
                <ul className="list-disc my-2" key={idx}>
                  <li>
                    <a href={val.link} target="_blank" rel="noopener noreferrer">
                      <ul className="list-disc my-2" key={idx}>
                        <li>
                          <span>
                            <a
                              href={val.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="font-bold"
                            >
                              <span>{val.name}</span>
                            </a>
                          </span>
                          <br />
                          {val.authors.map((author, idx) => (
                            <span
                              className={author === "Shenggui Li" ? "font-bold" : ""}
                              key={idx}
                            >
                              {author},&nbsp;
                            </span>
                          ))}
                          <br />
                        </li>
                      </ul>
                    </a>
                  </li>
                </ul>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default Techstack;
