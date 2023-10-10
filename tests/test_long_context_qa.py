from aleph_alpha_client import Client
from pytest import fixture
from intelligence_layer.long_context_qa import LongContextQa, LongContextQaInput

LONG_TEXT = """Robert Moses (December 18, 1888 – July 29, 1981) was an American urban planner and public official who worked in the New York metropolitan area during the early to mid 20th century. Despite never being elected to any office, Moses is regarded as one of the most powerful and influential individuals in the history of New York City and New York State. The grand scale of his infrastructural projects and his philosophy of urban development influenced a generation of engineers, architects, and urban planners across the United States.[2]

Moses held various positions throughout his more than forty-year long career. He at times held up to 12 titles simultaneously, including New York City Parks Commissioner and chairman of the Long Island State Park Commission.[3] Having worked closely with New York governor Al Smith early in his career, Moses became expert in writing laws and navigating and manipulating the inner workings of state government. He created and led numerous semi-autonomous public authorities, through which he controlled millions of dollars in revenue and directly issued bonds to fund new ventures with little outside input or oversight.

Moses's projects transformed the New York area and revolutionized the way cities in the U.S. were designed and built. As Long Island State Park Commissioner, Moses oversaw the construction of Jones Beach State Park, the most visited public beach in the United States,[4] and was the primary architect of the New York State Parkway System. As head of the Triborough Bridge Authority, Moses had near-complete control over bridges and tunnels in New York City as well as the tolls collected from them, and built, among others, the Triborough Bridge, the Brooklyn–Battery Tunnel, and the Throgs Neck Bridge, as well as several major highways. These roadways and bridges, alongside urban renewal efforts that saw the destruction of huge swaths of tenement housing and their replacement with large public housing projects, transformed the physical fabric of New York and inspired other cities to undertake similar development endeavors.

Moses's reputation declined following the publication of Robert Caro's Pulitzer Prize-winning biography The Power Broker (1974), which cast doubt on the purported benefits of many of Moses's projects and further cast Moses as racist. In large part because of The Power Broker, Moses is today considered a controversial figure in the history of New York City.

Early life and career
Moses was born in New Haven, Connecticut, on December 18, 1888, to German Jewish parents, Bella (Silverman) and Emanuel Moses.[5][6] He spent the first nine years of his life living at 83 Dwight Street in New Haven, two blocks from Yale University. In 1897, the Moses family moved to New York City,[7] where they lived on East 46th Street off Fifth Avenue.[8] Moses's father was a successful department store owner and real estate speculator in New Haven. In order for the family to move to New York City, he sold his real estate holdings and store, then retired.[7] Moses's mother was active in the settlement movement, with her own love of building. Robert Moses and his brother Paul attended several schools for their elementary and secondary education, including the Ethical Culture School, the Dwight School and the Mohegan Lake School, a military academy near Peekskill.[9]"""

@fixture
def qa(client: Client) -> LongContextQa:
    return LongContextQa(client, "info")


def test_qa_with_answer(qa: LongContextQa) -> None:
    
    question = "What is the name of the book about Robert Moses?"
    input = LongContextQaInput(text=LONG_TEXT, question=question)
    output = qa.run(input)
    assert output.answer
    assert "The Power Broker" in output.answer
    # highlights TODO


def test_qa_with_no_answer(qa: LongContextQa) -> None:
    question = "Who is the President of the united states?"
    input = LongContextQaInput(text=LONG_TEXT, question=question)
    output = qa.run(input)

    assert output.answer is None

