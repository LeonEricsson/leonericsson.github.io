import { publications } from "../../data";

function Techstack() {
  return (
    <div>
      {publications.map((publication, idx) => {
        return (
          <div className="text-black mb-1" key={idx}>
            <div className="mt-1 ml-8 my-2 flex flex-col">
              {publication.items.map((val, idx) => (
                <ul className="list-disc my-2 hover:-translate-y-1 hover:scale-102 transition duration-300" key={idx}>
                  <li>
                    <a href={val.url} target="_blank" rel="noopener noreferrer">
                      <ul className="list-disc my-2" key={idx}>
                        <li>
                          <span>
                              <span className="italic text-l ">{val.name}</span>
                          </span>
                          <br />
                          <div className="w-2/3">
                          {val.authors.map((author, idx) => (
                            <span
                              className="text-sm"
                              key={idx}
                            >
                              {author},&nbsp;
                            </span>
                          ))}
                          </div>
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
