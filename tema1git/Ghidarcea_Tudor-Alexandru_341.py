import glob
import requests
from bs4 import BeautifulSoup
import json
import re
import time

# functie care extrage link-ul catre pagina de recenzii pentru un film dat
def get_reviews_link(href):
    return f"https://www.imdb.com{href}reviews"

# functie pentru a extrage informatiile dintr-o recenzie
def parse_review(review):
    title = review.select("a.title")[0].text.strip()
    text = review.select("div.content")[0].text.strip()
    rating_element = review.select("span.rating-other-user-rating span")
    if len(rating_element) > 0:
        rating = int(rating_element[0].text)
    else:
        rating = None
    date = review.select("span.review-date")[0].text
    user = review.select("span.display-name-link a")[0].text
    return {"title": title, "text": text, "rating": rating, "date": date, "user": user}

# lista cu cele mai populare 250 de filme de pe IMDb
url = "https://www.imdb.com/chart/top/"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

movies = []

# pentru fiecare film identificam link-ul catre pagina sa de recenzii
for movie in soup.select("td.titleColumn"):
    title = movie.select("a")[0].text.strip()
    year_element = movie.select("span")[0].text.strip()
    year_match = re.search(r"\d{4}", year_element)
    if year_match:
        year = year_match.group()
    else:
        year = None
    reviews_link = get_reviews_link(movie.select_one("a")["href"])
    movies.append({"title": title, "year": year, "reviews_link": reviews_link})

# colectam date despre recenziile pentru fiecare film si le salvam intr-un fisier JSON
for movie in movies:
    print(f"Scraping reviews for {movie['title']} ({movie['year']})")
    reviews = []

    # extragem informatiile din primele recenzii
    response = requests.get(movie["reviews_link"])
    soup = BeautifulSoup(response.text, "html.parser")
    reviews.extend([parse_review(review) for review in soup.select("div.review-container")])

    # incercam sa extragem si restul recenziilor prin apasarea butonului "Load more"
    while True:
        try:
            load_more = soup.select("button.load-more-data")[0]
            pagination_key = load_more["data-key"]
            time.sleep(1)  # asteptam 1 secunda intre request-uri pentru a evita blocajele
            response = requests.get(f"{movie['reviews_link']}/_ajax?paginationKey={pagination_key}")
            soup = BeautifulSoup(response.json()["load_more_data"], "html.parser")
            reviews.extend([parse_review(review) for review in soup.select("div.review-container")])
        except:
            break

    # salvam recenziile intr-un fisier JSON
    title = movie['title'].replace("/", "-").replace(":", "-").strip()
    year = movie['year']
    filename = f"{title} ({year})"
    filename = re.sub(r'^a-zA-Z0-9\s()', '', filename).strip()
    filename = re.sub(r'\s+', ' ', filename)
    filename = re.sub(r'\s\(', ' (', filename)
    filename = re.sub(r'\)', ') ', filename)
    filename = re.sub(r'/', ' - ', filename)


    # daca filename-ul este gol dupa scoaterea caracterelor alfanumerice, dam un titlu default
    if not filename:
        filename = "Titlu Necunoscut"

    filename = f"{filename}.json"

    with open(filename, "w") as f:
        json.dump(reviews, f, indent=4)

    print(f"Collected {len(reviews)} reviews\n")


#ca sa avem si un dataset complet cu toate filmele la un loc,
#adaugam toate datele despre toate filmele intr-un singur fisier JSON, creand o noua coloana care
#sa reprezinte numele filmului despre care sunt datele ce urmeaza

#calea spre toate fișierele JSON create
json_files = glob.glob("*.json")

#cream o lista in care vom salva toate datele din fiecare fisier JSON
toate_datele = []

#parcurgem fiecare fisier JSON si adaugam in lista 'toate_datele' titlul si datele filmului
for file in json_files:
    with open(file, "r") as f:
        data = json.load(f)
        title = file.split(".")[0]
        for review in data:
            review["movie_title"] = title
        toate_datele.extend(data)

# cream un fisier JSON care sa contina datele din toate celelalte fisiere JSON
with open("toate_datele.json", "w") as f:
    json.dump(toate_datele, f, indent=4)
