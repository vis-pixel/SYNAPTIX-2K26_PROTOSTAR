#!/usr/bin/env python3
"""
BioRhythm X — Diet Plan Generator Test
Tests the full diet pipeline: BMR → TDEE → Macros → 7-day Meal Plan → Adaptive

Usage: python scripts/test_diet_plan.py
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.diet_engine.macro_calculator import calculate_macros, carb_cycling_plan
from app.diet_engine.meal_planner import MealPlanner, AdaptiveDietEngine, PRE_WORKOUT, POST_WORKOUT
from app.diet_engine.food_database import filter_foods


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def test_gym_mode():
    print_section("👤 PROFILE: 25M, 80kg, 180cm, Gym Rats Mode, Muscle Gain")
    macros = calculate_macros(
        weight_kg=80, height_cm=180, age=25, gender="male",
        activity_level="very_active", goal="muscle_gain"
    )
    print(f"  BMR:     {macros.bmr:.0f} kcal")
    print(f"  TDEE:    {macros.tdee:.0f} kcal")
    print(f"  Target:  {macros.calorie_target:.0f} kcal (surplus)")
    print(f"  Protein: {macros.protein_g:.0f}g ({macros.protein_pct:.0f}%)")
    print(f"  Carbs:   {macros.carbs_g:.0f}g ({macros.carbs_pct:.0f}%)")
    print(f"  Fat:     {macros.fat_g:.0f}g ({macros.fat_pct:.0f}%)")
    print(f"  Water:   {macros.water_ml:.0f}ml")

    carb_cycle = carb_cycling_plan(macros.carbs_g)
    print(f"\n  📅 Carb Cycle: {[f\"Day{c['day']}: {c['carbs_g']}g ({c['type']})\" for c in carb_cycle]}")

    planner = MealPlanner(macros, "gym_rats")
    day1 = planner.generate_day(1)
    print(f"\n  🍽️  Day 1 Meals:")
    for meal in day1:
        print(f"    [{meal['meal_time']}] {meal['meal_name']}: {meal['calories']:.0f}kcal | P:{meal['protein_g']:.0f}g C:{meal['carbs_g']:.0f}g F:{meal['fat_g']:.0f}g")
        for food in meal.get("food_items", [])[:2]:
            print(f"      • {food['food']} ({food['portion_g']:.0f}g) — {food['calories']:.0f}kcal")

    print(f"\n  Pre-Workout:  {PRE_WORKOUT['gym_rats']}")
    print(f"  Post-Workout: {POST_WORKOUT['gym_rats']}")


def test_fat_loss_mode():
    print_section("👤 PROFILE: 35F, 70kg, 165cm, Fat Loss Mode")
    macros = calculate_macros(
        weight_kg=70, height_cm=165, age=35, gender="female",
        activity_level="moderately_active", goal="fat_loss", body_fat_pct=30
    )
    print(f"  BMR:     {macros.bmr:.0f} kcal")
    print(f"  TDEE:    {macros.tdee:.0f} kcal")
    print(f"  Target:  {macros.calorie_target:.0f} kcal (deficit)")
    print(f"  Deficit: {macros.tdee - macros.calorie_target:.0f} kcal/day")
    print(f"  Protein: {macros.protein_g:.0f}g | Carbs: {macros.carbs_g:.0f}g | Fat: {macros.fat_g:.0f}g")


def test_adaptive_diet():
    print_section("⚙️  ADAPTIVE DIET ENGINE: Stressed + Poor Sleep Scenario")
    macros = calculate_macros(
        weight_kg=80, height_cm=180, age=25, gender="male",
        activity_level="very_active", goal="muscle_gain"
    )
    adaptive = AdaptiveDietEngine()
    adapted = adaptive.adapt(
        macros=macros,
        hrv=28.0,          # HRV dropped (baseline 55)
        baseline_hrv=55.0,
        stress_level=0.78, # High stress
        sleep_quality=0.45,# Poor sleep
        fatigue_load=72.0, # High fatigue
        illness_probability=0.52,
    )
    print(f"  Original Calories: {macros.calorie_target:.0f}")
    print(f"  Adapted Calories:  {adapted['calorie_target']:.0f}")
    print(f"  Adjusted Carbs:    {adapted['carbs_g']:.0f}g (was {macros.carbs_g:.0f}g)")
    print(f"  Adjusted Water:    {adapted['water_ml']:.0f}ml")
    print(f"\n  📋 Adjustments applied:")
    for reason in adapted["adjustment_reasons"]:
        print(f"    → {reason}")
    print(f"\n  🥗 Food Recommendations:")
    for food in adapted["food_recommendations"]:
        print(f"    → {food}")


def test_vegan_indian():
    print_section("👤 PROFILE: 28F, 55kg, 160cm, Indian Vegan Mode")
    macros = calculate_macros(
        weight_kg=55, height_cm=160, age=28, gender="female",
        activity_level="moderately_active", goal="general_health"
    )
    foods = filter_foods(["indian", "vegan"])
    print(f"  Indian Vegan foods available: {list(foods.keys())}")
    print(f"  Target: {macros.calorie_target:.0f} kcal | P:{macros.protein_g:.0f}g C:{macros.carbs_g:.0f}g F:{macros.fat_g:.0f}g")


if __name__ == "__main__":
    print("🍎 BioRhythm X — Diet Intelligence Test Suite")
    test_gym_mode()
    test_fat_loss_mode()
    test_adaptive_diet()
    test_vegan_indian()
    print(f"\n✅ All diet tests passed!")
