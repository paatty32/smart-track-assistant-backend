import logging

import httpx
from mcp.server import FastMCP

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from domain.TrainingPlan import TrainingPlan
from domain.TrainingPlanCreate import TrainingPlanCreate

URL = "https://api.open-meteo.com/v1/forecast"
params = {
    #Düsseldorf
	"latitude": 51.2217,
	"longitude": 6.7762,
	"daily": ["temperature_2m_max", "temperature_2m_min", "rain_sum", "snowfall_sum", "wind_speed_10m_max"],
	"hourly": ["temperature_2m", "rain", "showers", "wind_speed_10m"],
	"models": "icon_seamless",
	"timezone": "Europe/Berlin",
}

DATABASE_URL = "postgresql+asyncpg://user:password@localhost:5432/smartAssistantDb"
#Globale Enginge
#engine = create_engine(DATABASE_URL, echo=True)

enginne = create_async_engine(DATABASE_URL, echo=True)

#Async Session Factory
async_session = async_sessionmaker(enginne, class_=AsyncSession, expire_on_commit=False)

weather_mcp = FastMCP("weather")

logger = logging.getLogger(__name__)

def getWeather():
    response = httpx.get(URL, params=params)
    return response.text

async def insert_training_plan(session: AsyncSession, training_plan: TrainingPlanCreate):
    plan = TrainingPlan(**training_plan.model_dump())
    logger.info(f"Plan to add {plan}")
    session.add(plan)
    await session.commit()
    logger.info(f"End inserting training plan {training_plan}")
    await session.refresh(plan)
    return {
        "id": plan.id,
        "date": plan.datum,
        "weather": plan.weather,
    }


#TODO: tool bennen
@weather_mcp.tool()
def getWeatherTool():
    return getWeather()

@weather_mcp.tool()
async def insertTrainingPlan(training_plan: TrainingPlanCreate):

    async with async_session() as session:
        try:
            logger.info(f"Start inserting training plan {training_plan}")
            return await insert_training_plan(session, training_plan)
        except Exception:
            await session.rollback()
            raise
#
# @weather_mcp.tool()
# async def create_training_plan(
#     datum: date,
#     wetter: str,
#     aufwaermen: str,
#     hauptteil: str,
# ) -> TrainingPlanCreate:
#     plan = TrainingPlanCreate(
#         datum=datum,
#         wetter=wetter,
#         aufwaermen=aufwaermen,
#         hauptteil=hauptteil,
#     )
#     return plan



def main():
    weather_mcp.run(transport="streamable-http")

if __name__ == "__main__":
    main()