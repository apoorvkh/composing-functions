import requests

from . import CompositionalTask


def query_wikidata(sparql_query: str) -> list[tuple[str, str, str]]:
    response = requests.post(
        "https://qlever.cs.uni-freiburg.de/api/wikidata",
        headers={"Accept": "application/qlever-results+json", "Content-Type": "application/sparql-query"},
        data=sparql_query,
    )
    response.raise_for_status()
    return [tuple(s.strip('"') for s in r) for r in response.json()["res"]]


def tasks_from_wikidata_relations(hops: list[tuple[str, str, str]]) -> list[CompositionalTask]:
    return [CompositionalTask(x=x, Fx=Fx, Gx=None, GFx=GFx, FGx=None) for x, Fx, GFx in hops]


def book_author_birthyear(n: int = 2500) -> list[CompositionalTask]:
    return tasks_from_wikidata_relations(
        query_wikidata(f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wikibase: <http://wikiba.se/ontology#>
            PREFIX schema: <http://schema.org/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT DISTINCT
                (STR(?bookTitle) AS ?bookLabel)
                (STR(?authorName) AS ?authorLabel)
                (STR(YEAR(?birthDate)) AS ?birthYear)
            WHERE {{
                ?book wdt:P31 wd:Q7725634 ;
                    ^schema:about/wikibase:sitelinks ?sitelinks .
                ?book rdfs:label ?bookTitle .
                FILTER (LANG(?bookTitle) = "en")

                ?book wdt:P50 ?author .
                ?author rdfs:label ?authorName .
                FILTER (LANG(?authorName) = "en")

                ?author wdt:P569 ?birthDate .
                FILTER(YEAR(?birthDate) > 0)
            }}
            ORDER BY DESC(?sitelinks)
            LIMIT {n}
    """)
    )


def song_artist_birthyear(n: int = 1000) -> list[CompositionalTask]:
    return tasks_from_wikidata_relations(
        query_wikidata(f"""
            PREFIX schema: <http://schema.org/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX p: <http://www.wikidata.org/prop/>
            PREFIX ps: <http://www.wikidata.org/prop/statement/>
            PREFIX bd: <http://www.bigdata.com/rdf#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX wikibase: <http://wikiba.se/ontology#>

            SELECT DISTINCT
                (STR(?songLabel) AS ?songStr)
                (STR(?performerLabel) AS ?performerStr)
                (STR(YEAR(?birthDate)) AS ?birthYear)
            WHERE {{
                # Musical works / compositions
                ?song wdt:P31 wd:Q105543609 ;
                    ^schema:about/wikibase:sitelinks ?sitelinks .
                # Get primary performer
                {{
                    SELECT
                        ?song
                        (SAMPLE(?prefPerformer) AS ?prefPerformerSample)
                        (COUNT(DISTINCT ?prefPerformer) AS ?prefCount)
                        (SAMPLE(?normPerformer) AS ?normPerformerSample)
                        (COUNT(DISTINCT ?normPerformer) AS ?normCount)
                    WHERE {{
                        OPTIONAL {{
                            ?song p:P175 ?prefStmt .
                            ?prefStmt ps:P175 ?prefPerformer ;
                                    wikibase:rank wikibase:PreferredRank .
                        }}
                        OPTIONAL {{
                            ?song p:P175 ?normStmt .
                            ?normStmt ps:P175 ?normPerformer ;
                                    wikibase:rank wikibase:NormalRank .
                        }}
                    }}
                    GROUP BY ?song
                    HAVING ((?prefCount = 1) || (?prefCount = 0 && ?normCount = 1))
                }}
                BIND (COALESCE(?prefPerformerSample, ?normPerformerSample) AS ?performer)

                # Get the performer's birth date
                ?performer wdt:P569 ?birthDate .

                # Get song and performer labels
                ?song rdfs:label ?songLabel .
                FILTER (LANG(?songLabel) = "en")
                ?performer rdfs:label ?performerLabel .
                FILTER (LANG(?performerLabel) = "en")
            }}
            ORDER BY DESC(?sitelinks)
            LIMIT {n}
    """)
    )


def landmark_country_capital(min_sitelinks: int = 20) -> list[CompositionalTask]:
    def _query(landmarks_in_capital: bool = False) -> str:
        return f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wikibase: <http://wikiba.se/ontology#>
            PREFIX schema: <http://schema.org/>

            SELECT DISTINCT
                (STR(SAMPLE(?landmarkName)) AS ?landmarklabel)
                (STR(SAMPLE(?countryName)) AS ?countryLabel)
                (STR(SAMPLE(?capitalName)) AS ?capitalLabel)
            WHERE {{
                # landmark: tourist attractions or monument
                VALUES ?class {{ wd:Q570116 wd:Q4989906 }}
                ?landmark wdt:P31/wdt:P279* ?class ;
                            ^schema:about/wikibase:sitelinks ?sitelinks .
                ?landmark rdfs:label ?landmarkName .
                FILTER (LANG(?landmarkName) = "en") .

                # country
                ?landmark wdt:P17 ?country .
                ?country rdfs:label ?countryName .
                FILTER (LANG(?countryName) = "en")

                # capital
                ?country wdt:P36 ?capital .
                ?capital rdfs:label ?capitalName .
                FILTER (LANG(?capitalName) = "en")

                # optionally, keep only landmarks within the capital
                {"?landmark wdt:P131* ?capital ." if landmarks_in_capital else ""}
            }}
            GROUP BY ?landmark
            HAVING (
                COUNT(DISTINCT ?country) = 1
                && COUNT(DISTINCT ?capital) = 1
                && SAMPLE(?sitelinks) >= {min_sitelinks}
            )
            ORDER BY DESC(SAMPLE(?sitelinks))
        """

    all_landmarks = query_wikidata(_query())
    landmarks_in_capital = query_wikidata(_query(landmarks_in_capital=True))
    return tasks_from_wikidata_relations([lmk for lmk in all_landmarks if lmk not in landmarks_in_capital])


def park_country_capital(min_sitelinks: int = 20) -> list[CompositionalTask]:
    def _query(parks_in_capital: bool = False) -> str:
        return f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wikibase: <http://wikiba.se/ontology#>
            PREFIX schema: <http://schema.org/>

            SELECT DISTINCT
                (STR(SAMPLE(?parkName))    AS ?parkLabel)
                (STR(SAMPLE(?countryName)) AS ?countryLabel)
                (STR(SAMPLE(?capitalName)) AS ?capitalLabel)
            WHERE {{
                # park (and any subclass of it)
                VALUES ?class {{ wd:Q22698 }}
                ?park wdt:P31/wdt:P279* ?class ;
                      ^schema:about/wikibase:sitelinks ?sitelinks .
                ?park rdfs:label ?parkName .
                FILTER(LANG(?parkName) = "en") .

                # country that contains the park
                ?park wdt:P17 ?country .
                ?country rdfs:label ?countryName .
                FILTER(LANG(?countryName) = "en") .

                # capital of that country
                ?country wdt:P36 ?capital .
                ?capital rdfs:label ?capitalName .
                FILTER(LANG(?capitalName) = "en") .

                # optionally restrict to parks located within the capital city
                {"?park wdt:P131* ?capital ." if parks_in_capital else ""}
            }}
            GROUP BY ?park
            HAVING (
                COUNT(DISTINCT ?country) = 1
                && COUNT(DISTINCT ?capital) = 1
                && SAMPLE(?sitelinks) >= {min_sitelinks}
            )
            ORDER BY DESC(SAMPLE(?sitelinks))
        """

    all_parks = query_wikidata(_query())
    parks_in_capital = query_wikidata(_query(parks_in_capital=True))
    return tasks_from_wikidata_relations([p for p in all_parks if p not in parks_in_capital])


def movie_director_birthyear(min_votes: int = 100_000) -> list[CompositionalTask]:
    return tasks_from_wikidata_relations(
        query_wikidata(f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX imdb: <https://www.imdb.com/>

            SELECT DISTINCT
                (STR(?title) AS ?movieLabel)
                (STR(?directorName) AS ?directorLabel)
                (STR(YEAR(?birthDate)) AS ?birthYear)
            WHERE {{
                # Movies with exactly one director
                {{
                    SELECT ?movie
                    WHERE {{
                    ?movie wdt:P31 wd:Q11424 ;
                            wdt:P57  ?dirSub .
                    }}
                    GROUP BY ?movie
                    HAVING(COUNT(DISTINCT ?dirSub) = 1)
                }}

                # Director name and birth year
                ?movie wdt:P57     ?director ;
                        rdfs:label  ?title  FILTER(LANG(?title)="en") .

                ?director wdt:P569 ?birthDate ;
                            rdfs:label ?directorName  FILTER(LANG(?directorName)="en") .

                # Lookup IMDB votes
                ?movie wdt:P345 ?imdb_id ;
                SERVICE <https://qlever.cs.uni-freiburg.de/api/imdb> {{
                    ?movie_imdb imdb:id       ?imdb_id ;
                                imdb:type     "movie" ;
                                imdb:numVotes ?imdb_votes .
                }}
            }}
            GROUP BY ?movie ?title ?directorName ?birthDate ?imdb_votes
            HAVING(?imdb_votes >= {min_votes})
            ORDER BY DESC(?imdb_votes)
    """)
    )


def movie_company_founder(min_votes: int = 10_000) -> list[CompositionalTask]:
    return tasks_from_wikidata_relations(
        query_wikidata(f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX imdb: <https://www.imdb.com/>
            SELECT DISTINCT
                (STR(?title) AS ?movieLabel)
                (STR(?companyName) AS ?productionCompanyLabel)
                (STR(?founderName) AS ?founderLabel)
            WHERE {{
                ## 1) films with exactly one production company
                {{
                    SELECT ?movie WHERE {{
                    ?movie wdt:P31 wd:Q11424 ;
                            wdt:P272 ?company .
                    }}
                    GROUP BY ?movie
                    HAVING (COUNT(DISTINCT ?company) = 1)
                }}

                ## 2) title, company and IMDb ID
                ?movie rdfs:label ?title ;
                        wdt:P272 ?company ;
                        wdt:P345 ?imdb_id .
                FILTER (LANG(?title) = "en") .

                ## 3) Filter by votes
                SERVICE <https://qlever.cs.uni-freiburg.de/api/imdb> {{
                    ?movie_imdb imdb:id ?imdb_id ;
                                imdb:type "movie" ;
                                imdb:numVotes ?imdb_votes .
                }}
                FILTER (?imdb_votes >= {min_votes}) .

                ## 4) companies with exactly one founder
                {{
                    SELECT ?company WHERE {{
                    ?company wdt:P112 ?founder .
                    }}
                    GROUP BY ?company
                    HAVING (COUNT(DISTINCT ?founder) = 1)
                }}

                ## 5) company name, founder and birth date
                ?company rdfs:label ?companyName .
                FILTER (LANG(?companyName) = "en") .
                ?company wdt:P112 ?founder .
                ?founder rdfs:label ?founderName .
                FILTER (LANG(?founderName) = "en") .

                ## 6) ensure the founder is NOT credited on the film
                FILTER NOT EXISTS {{
                    # director, producer, screenwriter, executive producer
                    VALUES ?role {{ wdt:P57 wdt:P162 wdt:P58 wdt:P1431 }}
                    ?movie ?role ?founder .
                }}.
            }}
            ORDER BY DESC(?imdb_votes)
    """)
    )


def person_university_year(n: int = 5000) -> list[CompositionalTask]:
    return tasks_from_wikidata_relations(
        query_wikidata(f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX wikibase: <http://wikiba.se/ontology#>
            PREFIX schema: <http://schema.org/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX p: <http://www.wikidata.org/prop/>
            PREFIX ps: <http://www.wikidata.org/prop/statement/>
            PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
            SELECT DISTINCT
                (STR(SAMPLE(?personName)) AS ?personLabel)
                (STR(SAMPLE(?universityName)) AS ?universityLabel)
                (STR(YEAR(SAMPLE(?inception))) AS ?foundingYear)
            WHERE {{
                ?person wdt:P31 wd:Q5 ;
                        ^schema:about/wikibase:sitelinks ?sitelinks ;
                        p:P69 ?educated_at .
                ?educated_at ps:P69  ?university ;
                            pq:P512 ?degree .
                ?degree wdt:P31/wdt:P279* wd:Q163727 .  # bachelor's degrees
                ?university wdt:P571 ?inception .
                # labels
                ?person     rdfs:label    ?personName     FILTER(LANG(?personName)="en") .
                ?university rdfs:label    ?universityName FILTER(LANG(?universityName)="en") .
            }}
            GROUP BY ?person
            HAVING (COUNT(?university) = 1)
            ORDER BY DESC(SAMPLE(?sitelinks))
            LIMIT {n}
    """)
    )


def person_university_founder(n: int = 5000) -> list[CompositionalTask]:
    return tasks_from_wikidata_relations(
        query_wikidata(f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX wikibase: <http://wikiba.se/ontology#>
            PREFIX schema: <http://schema.org/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX p: <http://www.wikidata.org/prop/>
            PREFIX ps: <http://www.wikidata.org/prop/statement/>
            PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
            SELECT DISTINCT
            (STR(SAMPLE(?personName)) AS ?personLabel)
            (STR(SAMPLE(?universityName)) AS ?universityLabel)
            (STR(SAMPLE(?founderName)) AS ?founderLabel)
            WHERE {{
                ?person wdt:P31 wd:Q5 ;
                        ^schema:about/wikibase:sitelinks ?sitelinks ;
                        p:P69 ?educated_at .
                ?educated_at ps:P69  ?university ;
                            pq:P512 ?degree .
                ?degree wdt:P31/wdt:P279* wd:Q163727 .  # bachelor's degrees
                ?university wdt:P112 ?founder .
                # labels
                ?person     rdfs:label    ?personName     FILTER(LANG(?personName)="en") .
                ?university rdfs:label    ?universityName FILTER(LANG(?universityName)="en") .
                ?founder    rdfs:label    ?founderName    FILTER(LANG(?founderName)="en") .
            }}
            GROUP BY ?person
            HAVING ((COUNT(?university) = 1) && (COUNT(?founder) = 1))
            ORDER BY DESC(SAMPLE(?sitelinks))
            LIMIT {n}
    """)
    )


def product_company_ceo(min_sitelinks: int = 10) -> list[CompositionalTask]:
    return tasks_from_wikidata_relations(
        query_wikidata(f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX wikibase: <http://wikiba.se/ontology#>
            PREFIX schema: <http://schema.org/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX p: <http://www.wikidata.org/prop/>
            PREFIX ps: <http://www.wikidata.org/prop/statement/>

            SELECT DISTINCT
                (STR(SAMPLE(?productLabel)) AS ?productName)
                (STR(SAMPLE(?companyLabel)) AS ?companyName)
                (STR(SAMPLE(?ceoLabel)) AS ?ceoName)
            WHERE {{
                ## 1) Find each product and its single manufacturer
                ?product wdt:P31/wdt:P279* wd:Q2424752 ; # instance or subclass of “product”
                        ^schema:about/wikibase:sitelinks ?sitelinks ;
                        wdt:P176 ?company . # manufactured by

                ## 2) Sub-select to pick the one “primary” CEO for that company
                {{
                    SELECT
                        ?company
                        (SAMPLE(?prefCEO) AS ?prefCEOSample)
                        (COUNT(DISTINCT ?prefCEO) AS ?prefCount)
                        (SAMPLE(?normCEO) AS ?normCEOSample)
                        (COUNT(DISTINCT ?normCEO) AS ?normCount)
                    WHERE {{
                        OPTIONAL {{
                            ?company p:P169 ?stmtPref .
                            ?stmtPref ps:P169 ?prefCEO ; wikibase:rank wikibase:PreferredRank .
                        }}
                        OPTIONAL {{
                            ?company p:P169 ?stmtNorm .
                            ?stmtNorm ps:P169 ?normCEO ; wikibase:rank wikibase:NormalRank .
                        }}
                    }}
                    GROUP BY ?company
                    HAVING ((?prefCount = 1) || (?prefCount = 0 && ?normCount = 1))
                }}
                BIND (COALESCE(?prefCEOSample, ?normCEOSample) AS ?ceo)

                ## 3) Labels in English
                ?product rdfs:label ?productLabel FILTER (LANG(?productLabel) = "en") .
                ?company rdfs:label ?companyLabel FILTER (LANG(?companyLabel) = "en") .
                ?ceo rdfs:label ?ceoLabel FILTER (LANG(?ceoLabel) = "en") .
            }}
            GROUP BY ?product
            HAVING (COUNT(DISTINCT ?company) = 1 && COUNT(DISTINCT ?ceo) = 1 && SAMPLE(?sitelinks) >= {min_sitelinks})
            ORDER BY DESC(SAMPLE(?sitelinks))
    """)
    )


def product_company_hq(min_sitelinks: int = 20) -> list[CompositionalTask]:
    return tasks_from_wikidata_relations(
        query_wikidata(f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX wikibase: <http://wikiba.se/ontology#>
            PREFIX schema: <http://schema.org/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            SELECT DISTINCT
                (STR(SAMPLE(?productLabel)) AS ?productName)
                (STR(SAMPLE(?companyLabel)) AS ?companyName)
                (STR(SAMPLE(?hqLabel)) AS ?hqName)
            WHERE {{
                ## 1) Each product with exactly one manufacturer
                ?product wdt:P31/wdt:P279* wd:Q2424752 ; # instance/subclass of “product”
                        ^schema:about/wikibase:sitelinks ?sitelinks ;
                        wdt:P176 ?company . # manufactured by

                ## 2) That company's headquarters
                ?company wdt:P159 ?hq . # headquarters location

                ## 3) English labels
                ?product rdfs:label ?productLabel .
                FILTER (LANG(?productLabel) = "en")
                ?company rdfs:label ?companyLabel .
                FILTER (LANG(?companyLabel) = "en")
                ?hq rdfs:label ?hqLabel .
                FILTER (LANG(?hqLabel) = "en")
            }}
            GROUP BY ?product
            HAVING (COUNT(DISTINCT ?company) = 1 && COUNT(DISTINCT ?hq) = 1 && SAMPLE(?sitelinks) >= {min_sitelinks})
            ORDER BY DESC(SAMPLE(?sitelinks))
    """)
    )
