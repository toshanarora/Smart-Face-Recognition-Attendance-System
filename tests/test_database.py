import pytest

from src.exceptions import StudentAlreadyExistsError, StudentNotFoundError


def test_add_and_get_student(tmp_db):
    label = tmp_db.add_student("S001", "Aditi Sharma")
    assert label == 0

    student = tmp_db.get_student("S001")
    assert student is not None
    assert student.name == "Aditi Sharma"
    assert student.label_id == 0


def test_duplicate_enrollment_raises(tmp_db):
    tmp_db.add_student("S001", "Aditi Sharma")
    with pytest.raises(StudentAlreadyExistsError):
        tmp_db.add_student("S001", "Someone Else")


def test_label_ids_increment(tmp_db):
    l1 = tmp_db.add_student("S001", "A")
    l2 = tmp_db.add_student("S002", "B")
    l3 = tmp_db.add_student("S003", "C")
    assert [l1, l2, l3] == [0, 1, 2]


def test_list_students_sorted_by_name(tmp_db):
    tmp_db.add_student("S002", "Zara")
    tmp_db.add_student("S001", "Amit")
    names = [s.name for s in tmp_db.list_students()]
    assert names == ["Amit", "Zara"]


def test_mark_attendance_success(tmp_db):
    tmp_db.add_student("S001", "Amit")
    inserted = tmp_db.mark_attendance("S001", confidence=42.0)
    assert inserted is True

    records = tmp_db.get_attendance()
    assert len(records) == 1
    assert records[0].student_id == "S001"


def test_mark_attendance_once_per_day(tmp_db):
    tmp_db.add_student("S001", "Amit")
    first = tmp_db.mark_attendance("S001", confidence=42.0, once_per_day=True)
    second = tmp_db.mark_attendance("S001", confidence=50.0, once_per_day=True)
    assert first is True
    assert second is False  # duplicate same-day mark is silently ignored
    assert len(tmp_db.get_attendance()) == 1


def test_mark_attendance_unknown_student_raises(tmp_db):
    with pytest.raises(StudentNotFoundError):
        tmp_db.mark_attendance("GHOST", confidence=10.0)


def test_get_attendance_date_filter(tmp_db):
    tmp_db.add_student("S001", "Amit")
    tmp_db.mark_attendance("S001", confidence=10.0)
    today_records = tmp_db.get_attendance()
    assert len(today_records) == 1

    none_records = tmp_db.get_attendance("2000-01-01", "2000-01-02")
    assert len(none_records) == 0
