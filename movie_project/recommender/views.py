from django.shortcuts import render
from .logic import recommend


def home(request):
    recommendations = []
    movie_title = ""
    criterion = ""

    if request.method == "POST":
        movie_title = request.POST.get("movie_title", "")
        criterion = request.POST.get("criterion", "general")

        if movie_title:
            recommendations = recommend(movie_title, criterion)

    context = {
        "recommendations": recommendations,
        "movie_title": movie_title,
        "criterion": criterion,
        "criteria_list": ["general", "gen", "tema", "actori", "regizor"]
    }

    return render(request, "index.html", context)
