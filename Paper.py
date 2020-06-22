class Paper:
    def __init__(self, id):
        self.id = id
        self.test_cited_paper = []
        self.train_cited_paper = []

    def add_test_cited_paper(self, cited_paper_id):
        self.test_cited_paper.append(cited_paper_id)

    def add_train_cited_paper(self, cited_paper_id):
        self.train_cited_paper.append(cited_paper_id)