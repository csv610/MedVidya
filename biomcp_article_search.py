import subprocess
import datetime

class MedicalArticleSearch:
    def __init__(self):
        self.articles = []

    def search_articles(self, disease):
        self.articles = []

        if not disease.strip():
           return self.articles();

        try:
            result = subprocess.run(
                ["biomcp", "article", "search", "--disease", disease],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                raise Exception(f"Error running biomcp command: {result.stderr}")

            self.articles = self.parse_articles(result.stdout)
            return self.articles

        except Exception as e:
            raise e

    def parse_articles(self, output):

        records = output.strip().split("# Record ")
        articles = []

        for record in records[1:]:
            if not record.strip():
                continue  # Skip empty records
                
            article = {}
            lines = record.strip().splitlines()
            abstract_lines = []
            for line in lines:
                if line.startswith("Pmid:"):
                    article["pmid"] = line.replace("Pmid:", "").strip()
                elif line.startswith("Pmcid:"):
                    article["pmcid"] = line.replace("Pmcid:", "").strip()
                elif line.startswith("Title:"):
                    article["title"] = line.replace("Title:", "").strip()
                elif line.startswith("Journal:"):
                    article["journal"] = line.replace("Journal:", "").strip()
                elif line.startswith("Date:"):
                    article["date"] = line.replace("Date:", "").strip()
                elif line.startswith("Doi:"):
                    article["doi"] = line.replace("Doi:", "").strip()
                elif line.startswith("Abstract:"):
                    abstract_lines = []
                elif line.startswith("Pubmed Url:"):
                    article["pubmed_url"] = line.replace("Pubmed Url:", "").strip()
                elif line.startswith("Pmc Url:"):
                    article["pmc_url"] = line.replace("Pmc Url:", "").strip()
                elif line.startswith("Doi Url:"):
                    article["doi_url"] = line.replace("Doi Url:", "").strip()
                elif line.startswith("Authors:"):
                    authors_str = line.replace("Authors:", "").strip()
                    article["authors"] = [a.strip() for a in authors_str.split(",")]
                else:
                    abstract_lines.append(line.strip())

            article["abstract"] = " ".join(abstract_lines).strip()
            if article.get("title"):  # Only keep if title is present
                details = self.extract_article_details(article)
                article.update(details)  # Add formatted details to the article
                articles.append(article)

        return articles

    def get_article_count(self):
        """Get the number of articles found."""
        return len(self.articles)

    def extract_article_details(self, article):
        """Extract and format article details."""
        return {
            'title': article.get('title', 'N/A'),
            'authors': ", ".join(article.get('authors', [])),
            'journal': article.get('journal', 'N/A'),
            'date': article.get('date', 'N/A'),
            'year': article.get('date', 'N/A').split('-')[0] if '-' in article.get('date', 'N/A') else article.get('date', 'N/A').split(' ')[0] if ' ' in article.get('date', 'N/A') else article.get('date', 'N/A'),
            'pmid': article.get('pmid', 'N/A')
        }

    def get_article_citations(self):
        """Get formatted citations for all articles."""
        citations = []
        for i, article in enumerate(self.articles, 1):
            details = self.extract_article_details(article)
            citation = f"{i}. {details['title']}, {details['authors']}. {details['journal']}, {details['year']}. PMID: {details['pmid']}"
            citations.append(citation)
        
        return citations
