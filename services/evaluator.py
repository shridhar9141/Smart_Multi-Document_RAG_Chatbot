import re
from typing import List, Dict, Any
from chains.rag_chain import Level3RAGPipeline


EVALUATION_DATASET = [
    {
        "id": 1,
        "question": "What is Shridhar's email address?",
        "expected_answer": "shreeshridhar78@gmail.com",
        "ground_truth_keywords": ["shreeshridhar78@gmail.com"]
    },
    {
        "id": 2,
        "question": "What degree and branch is Shridhar pursuing?",
        "expected_answer": "Bachelor of Engineering (B.E.) in Electronics and Communication Engineering.",
        "ground_truth_keywords": ["Bachelor of Engineering", "Electronics and Communication Engineering"]
    },
    {
        "id": 3,
        "question": "What college is Shridhar studying at and what is his CGPA?",
        "expected_answer": "Kalpataru Institute of Technology, Tiptur with a CGPA of 8.35/10.",
        "ground_truth_keywords": ["Kalpataru Institute of Technology", "8.35"]
    },
    {
        "id": 4,
        "question": "What programming languages does Shridhar know?",
        "expected_answer": "Java, Python, C++, and JavaScript.",
        "ground_truth_keywords": ["Java", "Python", "C++", "JavaScript"]
    },
    {
        "id": 5,
        "question": "What database technologies are listed on his resume?",
        "expected_answer": "MongoDB and MySQL.",
        "ground_truth_keywords": ["MongoDB", "MySQL"]
    },
    {
        "id": 6,
        "question": "What was Shridhar's role and tech stack in the Student Result Management System project?",
        "expected_answer": "Role: Data & Java Developer. Tech Stack: Java, MySQL, SQL.",
        "ground_truth_keywords": ["Student Result Management System", "Data & Java Developer", "MySQL"]
    },
    {
        "id": 7,
        "question": "What was the group size and tech stack for the Task Management Application project?",
        "expected_answer": "Group size: 4. Tech stack: React.js, Node.js, Express.js, MongoDB, REST APIs.",
        "ground_truth_keywords": ["Task Management Application", "Group Size:4", "React.js", "Express.js"]
    },
    {
        "id": 8,
        "question": "What certifications has Shridhar earned?",
        "expected_answer": "SkillForge SQL Certification, Cisco Networking Academy Python Essentials, and Introduction to Modern AI.",
        "ground_truth_keywords": ["SkillForge", "Python Essentials", "Modern AI"]
    },
    {
        "id": 9,
        "question": "How many problems has Shridhar solved on GeeksforGeeks?",
        "expected_answer": "100+ Data Structures and Algorithms problems on GeeksforGeeks.",
        "ground_truth_keywords": ["100+", "GeeksforGeeks"]
    },
    {
        "id": 10,
        "question": "What tools and soft skills are mentioned on his resume?",
        "expected_answer": "Tools: Git, GitHub, VS Code, Postman. Soft Skills: Problem Solving, Communication, Analytical Thinking, Time Management, Quick Learning.",
        "ground_truth_keywords": ["Git", "Postman", "Problem Solving", "Communication"]
    }
]


class RAGEvaluator:

    @staticmethod
    def _is_doc_relevant(doc_content: str, keywords: List[str]) -> bool:
        doc_lower = doc_content.lower()
        return any(kw.lower() in doc_lower for kw in keywords)

    @staticmethod
    def _calculate_answer_correctness(generated_answer: str, expected_answer: str, keywords: List[str]) -> float:
        """
        Calculates Answer Correctness score (0.0 to 1.0) based on token overlap (F1 score)
        and ground truth keyword recall.
        """
        if not generated_answer or "couldn't find" in generated_answer.lower():
            return 0.0

        gen_lower = generated_answer.lower()

        # 1. Keyword Recall Score
        kw_matches = sum(1 for kw in keywords if kw.lower() in gen_lower)
        kw_score = kw_matches / len(keywords) if keywords else 1.0

        # 2. Token Overlap F1 Score
        gen_tokens = set(re.findall(r'\w+', gen_lower))
        exp_tokens = set(re.findall(r'\w+', expected_answer.lower()))

        intersection = gen_tokens.intersection(exp_tokens)
        if not intersection:
            f1_score = 0.0
        else:
            precision = len(intersection) / len(gen_tokens)
            recall = len(intersection) / len(exp_tokens)
            f1_score = 2 * (precision * recall) / (precision + recall)

        # Composite score: 60% keyword match + 40% F1 token overlap
        correctness = 0.6 * kw_score + 0.4 * f1_score
        return round(min(1.0, correctness), 4)

    @classmethod
    def evaluate(cls, pipeline: Level3RAGPipeline, dataset: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        if dataset is None:
            dataset = EVALUATION_DATASET

        results = []
        hits = 0
        total_precision = 0.0
        total_recall = 0.0
        total_correctness = 0.0

        for item in dataset:
            q_id = item["id"]
            question = item["question"]
            expected_ans = item["expected_answer"]
            gt_keywords = item["ground_truth_keywords"]

            # Run query through Level 3 RAG Pipeline
            output = pipeline.invoke({"question": question, "chat_history": []})
            generated_ans = output.get("answer", "")
            retrieved_docs = output.get("source_documents", [])  # Top 3 reranked docs

            # Relevant retrieved docs count
            relevant_retrieved = [
                doc for doc in retrieved_docs
                if cls._is_doc_relevant(doc.page_content, gt_keywords)
            ]

            # 1. Retrieval Accuracy (Hit Rate)
            hit = len(relevant_retrieved) > 0
            if hit:
                hits += 1

            # 2. Precision
            precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0.0
            total_precision += precision

            # 3. Recall
            recall = 1.0 if hit else 0.0
            total_recall += recall

            # 4. Answer Correctness
            correctness = cls._calculate_answer_correctness(generated_ans, expected_ans, gt_keywords)
            total_correctness += correctness

            results.append({
                "id": q_id,
                "question": question,
                "expected_answer": expected_ans,
                "generated_answer": generated_ans,
                "hit": hit,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "answer_correctness": round(correctness, 4),
                "retrieved_count": len(retrieved_docs),
                "relevant_retrieved_count": len(relevant_retrieved)
            })

        n = len(dataset)
        summary = {
            "total_questions": n,
            "retrieval_accuracy": round((hits / n) * 100, 2),  # Percentage
            "average_precision": round(total_precision / n, 4),
            "average_recall": round(total_recall / n, 4),
            "average_answer_correctness": round(total_correctness / n, 4),
            "per_question_results": results
        }

        return summary
