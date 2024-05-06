"""
Prepare data for annotation relating to education and crisis together in the
proceedings.

This is intended to be a spreadsheet for annotation of how crisis discourse is employed
relating to education, and in particular how that is deployed in speeches about
education compared to mentions in speeches on other topics.

One concrete example: discussions about legislation relating to e-cigarettes often
includes mention of schools and students as a particular problem, but these speeches
are not *about* education.
 
These script assumes that the index and corpus have already been created by running:

python hansard_corpus.py index cluster

"""

import concurrent.futures as cf
import csv
import multiprocessing as mp

import lxml

from hyperreal.index import Index
from hansard_corpus import HansardCorpus

db = "tidy_hansard.db"
db_index = "tidy_hansard_index.db"

if __name__ == "__main__":

    corpus = HansardCorpus(db)

    mp_context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(mp_context=mp_context) as pool:
        idx = Index(db_index, corpus=corpus, pool=pool)

        crisis_terms = ["crisis", "crises"]
        # These were derived from the feature_clustering model, but are baked in here
        # for transparency and preservation of the method used. The commented out
        # words were manually reviewed and removed.
        education_terms = [
            "education",
            "training",
            "school",
            "schools",
            "skills",
            # Youth matches on many non-educational issues ("crime", "health")
            # "youth",
            "educational",
            "trained",
            "learning",
            "educated",
            "taught",
            "educate",
            "teach",
            "placement",
            "principals",
            "educating",
            "leavers",
            "boarding",
            "retraining",
            "placements",
            "mentoring",
            "schoolchildren",
            "untrained",
            "vocation",
            "trainers",
            "excursions",
            "induction",
            "learners",
            "aides",
            "civics",
            "cyss",
            "gymnasium",
            "schoolteacher",
            "retrained",
            "leaver",
            "uneducated",
            "clontarf",
            "schoolteachers",
            "learner",
            "prep",
            "indoctrination",
            "fete",
            "scholastic",
            "interpersonal",
            "boarders",
            "educates",
            "vacations",
            "equips",
            "eduction",
            "tuckshop",
            "aptitudes",
            "boarder",
            "university",
            "students",
            "student",
            "universities",
            "teachers",
            "tertiary",
            "teaching",
            "academic",
            "teacher",
            "enrolled",
            "enrolment",
            "schooling",
            "enrolments",
            "monash",
            "curriculum",
            "chancellor",
            "austudy",
            "enrol",
            "classroom",
            "classrooms",
            "anu",
            "gonski",
            "faculty",
            "lecturer",
            "professors",
            "pupils",
            "school's",
            "uni",
            "abstudy",
            "enrolling",
            "phd",
            "teas",
            "pupil",
            "textbooks",
            "karmel",
            "harvard",
            "educationally",
            "headmaster",
            "naplan",
            "doctorate",
            "teacher's",
            "classmates",
            "unsw",
            "qut",
            "extracurricular",
            "aeu",
            "headmasters",
            "pisa",
            "yale",
            "curriculums",
            "acara",
            "vitae",
            "college",
            "courses",
            "colleges",
            "graduate",
            "graduates",
            "campus",
            "tuition",
            "campuses",
            "bachelor",
            "grammar",
            "diploma",
            "university's",
            "faculties",
            "graduation",
            "duntroon",
            "graduating",
            "student's",
            "lecturers",
            "curricula",
            "trinity",
            "diplomas",
            "matriculation",
            "academically",
            "tutorial",
            "exams",
            "swinburne",
            "tutors",
            "exam",
            "esos",
            "rmit",
            "hsc",
            "tutor",
            "accrediting",
            "refresher",
            "educationists",
            "alumni",
            "education's",
            "college's",
            "syllabus",
            "deans",
            "bachelors",
            "amc",
            "tutoring",
            "professorial",
            "cae",
            "caes",
            "marist",
            "educationalists",
            "radford",
            "mceetya",
            "vce",
            "ph.d",
            "graduations",
            "divinity",
            "b.a",
            "dropout",
            "matriculated",
            "roseworthy",
            "cricos",
            "elicos",
            "ctec",
            "avondale",
            "dux",
            "uq",
            "assertiveness",
            "unis",
            "atar",
        ]

        matching_docs = idx.field_proximity_query(
            "text", [crisis_terms, education_terms], 10
        )
        print(len(matching_docs))

        converter = idx.field_values["text"].segment_to_str

        window_size = 12

        full_docs = {key: doc for _, key, doc in idx.docs(matching_docs)}

        with open("education_crisis_speeches.csv", "w") as f:

            fields = [
                "url",
                "date",
                "house",
                "title",
                "matched_words",
                "speech_about_education",
                "any_snippet_about_education",
                "snippets",
                "text",
            ]

            writer = csv.DictWriter(
                f, fields, quoting=csv.QUOTE_ALL, extrasaction="ignore"
            )
            writer.writeheader()

            for _, key, doc_features in idx.indexable_docs(matching_docs):

                doc = full_docs[key]

                feature_matches = idx.match_doc_features(
                    doc_features, {"text": crisis_terms}
                )

                all_matches = idx.match_doc_features(
                    doc_features, {"text": education_terms}
                )

                positions = sorted(
                    p
                    for positions in feature_matches["text"].values()
                    for p in positions
                )

                snippets = "\n\n".join(
                    converter(
                        doc_features["text"],
                        max(p - window_size, 0),
                        p + window_size + 1,
                        highlight=crisis_terms + education_terms,
                    )
                    for p in positions
                )

                doc["snippets"] = snippets
                # Handle excel limitations on the size of a single cell...
                doc["text"] = corpus.doc_to_str(doc)
                if len(doc["text"]) >= 32000:
                    doc["text"] = doc["text"][:32000] + " [...]"
                doc["matched_words"] = ", ".join(all_matches["text"].keys())
                doc["speech_about_education"] = ""
                doc["any_snippet_about_education"] = ""
                writer.writerow(doc)
