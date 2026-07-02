from app import (
    calculate_stylometric_signal,
    classify_with_llm,
    combine_signal_scores,
    score_to_attribution,
)


samples = {
    "clearly_ai": (
        "Artificial intelligence represents a transformative paradigm shift in modern society. "
        "It is important to note that while the benefits of AI are numerous, it is equally "
        "essential to consider the ethical implications. Furthermore, stakeholders across "
        "various sectors must collaborate to ensure responsible deployment."
    ),
    "clearly_human": (
        "ok so i finally tried that new ramen place downtown and honestly? "
        "underwhelming. the broth was fine but they put WAY too much sodium in it and "
        "i was thirsty for like three hours after. my friend got the spicy version and "
        "said it was better. probably won't go back unless someone drags me there"
    ),
    "borderline_formal_human": (
        "The relationship between monetary policy and asset price inflation has been "
        "extensively studied in the literature. Central banks face a fundamental tension "
        "between their mandate for price stability and the unintended consequences of "
        "prolonged low interest rates on equity and real estate valuations."
    ),
    "borderline_edited_ai": (
        "I've been thinking a lot about remote work lately. There are genuine tradeoffs — "
        "flexibility and no commute on one side, isolation and blurred work-life boundaries "
        "on the other. Studies show productivity varies widely by individual and role type."
    ),
}


for name, text in samples.items():
    print("=" * 80)
    print(name)

    llm_result = classify_with_llm(text)
    stylometric_result = calculate_stylometric_signal(text)

    combined_score = combine_signal_scores(
        llm_score=llm_result["score"],
        stylometric_score=stylometric_result["score"],
    )

    attribution = score_to_attribution(combined_score)

    print("LLM score:", llm_result["score"])
    print("LLM reason:", llm_result["reason"])
    print("Stylometric score:", stylometric_result["score"])
    print("Stylometric features:", stylometric_result["features"])
    print("Combined score:", combined_score)
    print("Attribution:", attribution)