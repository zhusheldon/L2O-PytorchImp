// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: gps.proto

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "gps.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)

namespace gps {

namespace {

const ::google::protobuf::Descriptor* Sample_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  Sample_reflection_ = NULL;
const ::google::protobuf::EnumDescriptor* SampleType_descriptor_ = NULL;

}  // namespace


void protobuf_AssignDesc_gps_2eproto() GOOGLE_ATTRIBUTE_COLD;
void protobuf_AssignDesc_gps_2eproto() {
  protobuf_AddDesc_gps_2eproto();
  const ::google::protobuf::FileDescriptor* file =
    ::google::protobuf::DescriptorPool::generated_pool()->FindFileByName(
      "gps.proto");
  GOOGLE_CHECK(file != NULL);
  Sample_descriptor_ = file->message_type(0);
  static const int Sample_offsets_[8] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Sample, t_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Sample, dx_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Sample, du_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Sample, do__),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Sample, x_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Sample, u_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Sample, obs_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Sample, meta_),
  };
  Sample_reflection_ =
    ::google::protobuf::internal::GeneratedMessageReflection::NewGeneratedMessageReflection(
      Sample_descriptor_,
      Sample::internal_default_instance(),
      Sample_offsets_,
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Sample, _has_bits_),
      -1,
      -1,
      sizeof(Sample),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Sample, _internal_metadata_));
  SampleType_descriptor_ = file->enum_type(0);
}

namespace {

GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_AssignDescriptors_once_);
void protobuf_AssignDescriptorsOnce() {
  ::google::protobuf::GoogleOnceInit(&protobuf_AssignDescriptors_once_,
                 &protobuf_AssignDesc_gps_2eproto);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
      Sample_descriptor_, Sample::internal_default_instance());
}

}  // namespace

void protobuf_ShutdownFile_gps_2eproto() {
  Sample_default_instance_.Shutdown();
  delete Sample_reflection_;
}

void protobuf_InitDefaults_gps_2eproto_impl() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  Sample_default_instance_.DefaultConstruct();
  Sample_default_instance_.get_mutable()->InitAsDefaultInstance();
}

GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_InitDefaults_gps_2eproto_once_);
void protobuf_InitDefaults_gps_2eproto() {
  ::google::protobuf::GoogleOnceInit(&protobuf_InitDefaults_gps_2eproto_once_,
                 &protobuf_InitDefaults_gps_2eproto_impl);
}
void protobuf_AddDesc_gps_2eproto_impl() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  protobuf_InitDefaults_gps_2eproto();
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
    "\n\tgps.proto\022\003gps\"x\n\006Sample\022\t\n\001T\030\001 \001(\r\022\n\n"
    "\002dX\030\002 \001(\r\022\n\n\002dU\030\003 \001(\r\022\n\n\002dO\030\004 \001(\r\022\r\n\001X\030\005"
    " \003(\002B\002\020\001\022\r\n\001U\030\006 \003(\002B\002\020\001\022\017\n\003obs\030\007 \003(\002B\002\020\001"
    "\022\020\n\004meta\030\010 \003(\002B\002\020\001*q\n\nSampleType\022\n\n\006ACTI"
    "ON\020\000\022\013\n\007CUR_LOC\020\001\022\027\n\023PAST_OBJ_VAL_DELTAS"
    "\020\002\022\016\n\nPAST_GRADS\020\003\022\014\n\010CUR_GRAD\020\004\022\023\n\017PAST"
    "_LOC_DELTAS\020\005", 253);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "gps.proto", &protobuf_RegisterTypes);
  ::google::protobuf::internal::OnShutdown(&protobuf_ShutdownFile_gps_2eproto);
}

GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_AddDesc_gps_2eproto_once_);
void protobuf_AddDesc_gps_2eproto() {
  ::google::protobuf::GoogleOnceInit(&protobuf_AddDesc_gps_2eproto_once_,
                 &protobuf_AddDesc_gps_2eproto_impl);
}
// Force AddDescriptors() to be called at static initialization time.
struct StaticDescriptorInitializer_gps_2eproto {
  StaticDescriptorInitializer_gps_2eproto() {
    protobuf_AddDesc_gps_2eproto();
  }
} static_descriptor_initializer_gps_2eproto_;
const ::google::protobuf::EnumDescriptor* SampleType_descriptor() {
  protobuf_AssignDescriptorsOnce();
  return SampleType_descriptor_;
}
bool SampleType_IsValid(int value) {
  switch (value) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
      return true;
    default:
      return false;
  }
}


namespace {

static void MergeFromFail(int line) GOOGLE_ATTRIBUTE_COLD GOOGLE_ATTRIBUTE_NORETURN;
static void MergeFromFail(int line) {
  ::google::protobuf::internal::MergeFromFail(__FILE__, line);
}

}  // namespace


// ===================================================================

#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int Sample::kTFieldNumber;
const int Sample::kDXFieldNumber;
const int Sample::kDUFieldNumber;
const int Sample::kDOFieldNumber;
const int Sample::kXFieldNumber;
const int Sample::kUFieldNumber;
const int Sample::kObsFieldNumber;
const int Sample::kMetaFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

Sample::Sample()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  if (this != internal_default_instance()) protobuf_InitDefaults_gps_2eproto();
  SharedCtor();
  // @@protoc_insertion_point(constructor:gps.Sample)
}

void Sample::InitAsDefaultInstance() {
}

Sample::Sample(const Sample& from)
  : ::google::protobuf::Message(),
    _internal_metadata_(NULL) {
  SharedCtor();
  UnsafeMergeFrom(from);
  // @@protoc_insertion_point(copy_constructor:gps.Sample)
}

void Sample::SharedCtor() {
  _cached_size_ = 0;
  ::memset(&t_, 0, reinterpret_cast<char*>(&do__) -
    reinterpret_cast<char*>(&t_) + sizeof(do__));
}

Sample::~Sample() {
  // @@protoc_insertion_point(destructor:gps.Sample)
  SharedDtor();
}

void Sample::SharedDtor() {
}

void Sample::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* Sample::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return Sample_descriptor_;
}

const Sample& Sample::default_instance() {
  protobuf_InitDefaults_gps_2eproto();
  return *internal_default_instance();
}

::google::protobuf::internal::ExplicitlyConstructed<Sample> Sample_default_instance_;

Sample* Sample::New(::google::protobuf::Arena* arena) const {
  Sample* n = new Sample;
  if (arena != NULL) {
    arena->Own(n);
  }
  return n;
}

void Sample::Clear() {
// @@protoc_insertion_point(message_clear_start:gps.Sample)
#if defined(__clang__)
#define ZR_HELPER_(f) \
  _Pragma("clang diagnostic push") \
  _Pragma("clang diagnostic ignored \"-Winvalid-offsetof\"") \
  __builtin_offsetof(Sample, f) \
  _Pragma("clang diagnostic pop")
#else
#define ZR_HELPER_(f) reinterpret_cast<char*>(\
  &reinterpret_cast<Sample*>(16)->f)
#endif

#define ZR_(first, last) do {\
  ::memset(&(first), 0,\
           ZR_HELPER_(last) - ZR_HELPER_(first) + sizeof(last));\
} while (0)

  ZR_(t_, do__);

#undef ZR_HELPER_
#undef ZR_

  x_.Clear();
  u_.Clear();
  obs_.Clear();
  meta_.Clear();
  _has_bits_.Clear();
  if (_internal_metadata_.have_unknown_fields()) {
    mutable_unknown_fields()->Clear();
  }
}

bool Sample::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:gps.Sample)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoff(127);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // optional uint32 T = 1;
      case 1: {
        if (tag == 8) {
          set_has_t();
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint32, ::google::protobuf::internal::WireFormatLite::TYPE_UINT32>(
                 input, &t_)));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(16)) goto parse_dX;
        break;
      }

      // optional uint32 dX = 2;
      case 2: {
        if (tag == 16) {
         parse_dX:
          set_has_dx();
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint32, ::google::protobuf::internal::WireFormatLite::TYPE_UINT32>(
                 input, &dx_)));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(24)) goto parse_dU;
        break;
      }

      // optional uint32 dU = 3;
      case 3: {
        if (tag == 24) {
         parse_dU:
          set_has_du();
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint32, ::google::protobuf::internal::WireFormatLite::TYPE_UINT32>(
                 input, &du_)));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(32)) goto parse_dO;
        break;
      }

      // optional uint32 dO = 4;
      case 4: {
        if (tag == 32) {
         parse_dO:
          set_has_do_();
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint32, ::google::protobuf::internal::WireFormatLite::TYPE_UINT32>(
                 input, &do__)));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(42)) goto parse_X;
        break;
      }

      // repeated float X = 5 [packed = true];
      case 5: {
        if (tag == 42) {
         parse_X:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPackedPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, this->mutable_x())));
        } else if (tag == 45) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadRepeatedPrimitiveNoInline<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 1, 42, input, this->mutable_x())));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(50)) goto parse_U;
        break;
      }

      // repeated float U = 6 [packed = true];
      case 6: {
        if (tag == 50) {
         parse_U:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPackedPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, this->mutable_u())));
        } else if (tag == 53) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadRepeatedPrimitiveNoInline<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 1, 50, input, this->mutable_u())));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(58)) goto parse_obs;
        break;
      }

      // repeated float obs = 7 [packed = true];
      case 7: {
        if (tag == 58) {
         parse_obs:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPackedPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, this->mutable_obs())));
        } else if (tag == 61) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadRepeatedPrimitiveNoInline<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 1, 58, input, this->mutable_obs())));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(66)) goto parse_meta;
        break;
      }

      // repeated float meta = 8 [packed = true];
      case 8: {
        if (tag == 66) {
         parse_meta:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPackedPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, this->mutable_meta())));
        } else if (tag == 69) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadRepeatedPrimitiveNoInline<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 1, 66, input, this->mutable_meta())));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectAtEnd()) goto success;
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0 ||
            ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:gps.Sample)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:gps.Sample)
  return false;
#undef DO_
}

void Sample::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:gps.Sample)
  // optional uint32 T = 1;
  if (has_t()) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt32(1, this->t(), output);
  }

  // optional uint32 dX = 2;
  if (has_dx()) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt32(2, this->dx(), output);
  }

  // optional uint32 dU = 3;
  if (has_du()) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt32(3, this->du(), output);
  }

  // optional uint32 dO = 4;
  if (has_do_()) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt32(4, this->do_(), output);
  }

  // repeated float X = 5 [packed = true];
  if (this->x_size() > 0) {
    ::google::protobuf::internal::WireFormatLite::WriteTag(5, ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED, output);
    output->WriteVarint32(_x_cached_byte_size_);
  }
  for (int i = 0; i < this->x_size(); i++) {
    ::google::protobuf::internal::WireFormatLite::WriteFloatNoTag(
      this->x(i), output);
  }

  // repeated float U = 6 [packed = true];
  if (this->u_size() > 0) {
    ::google::protobuf::internal::WireFormatLite::WriteTag(6, ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED, output);
    output->WriteVarint32(_u_cached_byte_size_);
  }
  for (int i = 0; i < this->u_size(); i++) {
    ::google::protobuf::internal::WireFormatLite::WriteFloatNoTag(
      this->u(i), output);
  }

  // repeated float obs = 7 [packed = true];
  if (this->obs_size() > 0) {
    ::google::protobuf::internal::WireFormatLite::WriteTag(7, ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED, output);
    output->WriteVarint32(_obs_cached_byte_size_);
  }
  for (int i = 0; i < this->obs_size(); i++) {
    ::google::protobuf::internal::WireFormatLite::WriteFloatNoTag(
      this->obs(i), output);
  }

  // repeated float meta = 8 [packed = true];
  if (this->meta_size() > 0) {
    ::google::protobuf::internal::WireFormatLite::WriteTag(8, ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED, output);
    output->WriteVarint32(_meta_cached_byte_size_);
  }
  for (int i = 0; i < this->meta_size(); i++) {
    ::google::protobuf::internal::WireFormatLite::WriteFloatNoTag(
      this->meta(i), output);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:gps.Sample)
}

::google::protobuf::uint8* Sample::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:gps.Sample)
  // optional uint32 T = 1;
  if (has_t()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt32ToArray(1, this->t(), target);
  }

  // optional uint32 dX = 2;
  if (has_dx()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt32ToArray(2, this->dx(), target);
  }

  // optional uint32 dU = 3;
  if (has_du()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt32ToArray(3, this->du(), target);
  }

  // optional uint32 dO = 4;
  if (has_do_()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt32ToArray(4, this->do_(), target);
  }

  // repeated float X = 5 [packed = true];
  if (this->x_size() > 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteTagToArray(
      5,
      ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED,
      target);
    target = ::google::protobuf::io::CodedOutputStream::WriteVarint32ToArray(
      _x_cached_byte_size_, target);
  }
  for (int i = 0; i < this->x_size(); i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteFloatNoTagToArray(this->x(i), target);
  }

  // repeated float U = 6 [packed = true];
  if (this->u_size() > 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteTagToArray(
      6,
      ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED,
      target);
    target = ::google::protobuf::io::CodedOutputStream::WriteVarint32ToArray(
      _u_cached_byte_size_, target);
  }
  for (int i = 0; i < this->u_size(); i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteFloatNoTagToArray(this->u(i), target);
  }

  // repeated float obs = 7 [packed = true];
  if (this->obs_size() > 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteTagToArray(
      7,
      ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED,
      target);
    target = ::google::protobuf::io::CodedOutputStream::WriteVarint32ToArray(
      _obs_cached_byte_size_, target);
  }
  for (int i = 0; i < this->obs_size(); i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteFloatNoTagToArray(this->obs(i), target);
  }

  // repeated float meta = 8 [packed = true];
  if (this->meta_size() > 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteTagToArray(
      8,
      ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED,
      target);
    target = ::google::protobuf::io::CodedOutputStream::WriteVarint32ToArray(
      _meta_cached_byte_size_, target);
  }
  for (int i = 0; i < this->meta_size(); i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteFloatNoTagToArray(this->meta(i), target);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:gps.Sample)
  return target;
}

size_t Sample::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:gps.Sample)
  size_t total_size = 0;

  if (_has_bits_[0 / 32] & 15u) {
    // optional uint32 T = 1;
    if (has_t()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::UInt32Size(
          this->t());
    }

    // optional uint32 dX = 2;
    if (has_dx()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::UInt32Size(
          this->dx());
    }

    // optional uint32 dU = 3;
    if (has_du()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::UInt32Size(
          this->du());
    }

    // optional uint32 dO = 4;
    if (has_do_()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::UInt32Size(
          this->do_());
    }

  }
  // repeated float X = 5 [packed = true];
  {
    size_t data_size = 0;
    unsigned int count = this->x_size();
    data_size = 4UL * count;
    if (data_size > 0) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(data_size);
    }
    int cached_size = ::google::protobuf::internal::ToCachedSize(data_size);
    GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
    _x_cached_byte_size_ = cached_size;
    GOOGLE_SAFE_CONCURRENT_WRITES_END();
    total_size += data_size;
  }

  // repeated float U = 6 [packed = true];
  {
    size_t data_size = 0;
    unsigned int count = this->u_size();
    data_size = 4UL * count;
    if (data_size > 0) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(data_size);
    }
    int cached_size = ::google::protobuf::internal::ToCachedSize(data_size);
    GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
    _u_cached_byte_size_ = cached_size;
    GOOGLE_SAFE_CONCURRENT_WRITES_END();
    total_size += data_size;
  }

  // repeated float obs = 7 [packed = true];
  {
    size_t data_size = 0;
    unsigned int count = this->obs_size();
    data_size = 4UL * count;
    if (data_size > 0) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(data_size);
    }
    int cached_size = ::google::protobuf::internal::ToCachedSize(data_size);
    GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
    _obs_cached_byte_size_ = cached_size;
    GOOGLE_SAFE_CONCURRENT_WRITES_END();
    total_size += data_size;
  }

  // repeated float meta = 8 [packed = true];
  {
    size_t data_size = 0;
    unsigned int count = this->meta_size();
    data_size = 4UL * count;
    if (data_size > 0) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(data_size);
    }
    int cached_size = ::google::protobuf::internal::ToCachedSize(data_size);
    GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
    _meta_cached_byte_size_ = cached_size;
    GOOGLE_SAFE_CONCURRENT_WRITES_END();
    total_size += data_size;
  }

  if (_internal_metadata_.have_unknown_fields()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        unknown_fields());
  }
  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = cached_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void Sample::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:gps.Sample)
  if (GOOGLE_PREDICT_FALSE(&from == this)) MergeFromFail(__LINE__);
  const Sample* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const Sample>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:gps.Sample)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:gps.Sample)
    UnsafeMergeFrom(*source);
  }
}

void Sample::MergeFrom(const Sample& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:gps.Sample)
  if (GOOGLE_PREDICT_TRUE(&from != this)) {
    UnsafeMergeFrom(from);
  } else {
    MergeFromFail(__LINE__);
  }
}

void Sample::UnsafeMergeFrom(const Sample& from) {
  GOOGLE_DCHECK(&from != this);
  x_.UnsafeMergeFrom(from.x_);
  u_.UnsafeMergeFrom(from.u_);
  obs_.UnsafeMergeFrom(from.obs_);
  meta_.UnsafeMergeFrom(from.meta_);
  if (from._has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    if (from.has_t()) {
      set_t(from.t());
    }
    if (from.has_dx()) {
      set_dx(from.dx());
    }
    if (from.has_du()) {
      set_du(from.du());
    }
    if (from.has_do_()) {
      set_do_(from.do_());
    }
  }
  if (from._internal_metadata_.have_unknown_fields()) {
    ::google::protobuf::UnknownFieldSet::MergeToInternalMetdata(
      from.unknown_fields(), &_internal_metadata_);
  }
}

void Sample::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:gps.Sample)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Sample::CopyFrom(const Sample& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:gps.Sample)
  if (&from == this) return;
  Clear();
  UnsafeMergeFrom(from);
}

bool Sample::IsInitialized() const {

  return true;
}

void Sample::Swap(Sample* other) {
  if (other == this) return;
  InternalSwap(other);
}
void Sample::InternalSwap(Sample* other) {
  std::swap(t_, other->t_);
  std::swap(dx_, other->dx_);
  std::swap(du_, other->du_);
  std::swap(do__, other->do__);
  x_.UnsafeArenaSwap(&other->x_);
  u_.UnsafeArenaSwap(&other->u_);
  obs_.UnsafeArenaSwap(&other->obs_);
  meta_.UnsafeArenaSwap(&other->meta_);
  std::swap(_has_bits_[0], other->_has_bits_[0]);
  _internal_metadata_.Swap(&other->_internal_metadata_);
  std::swap(_cached_size_, other->_cached_size_);
}

::google::protobuf::Metadata Sample::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = Sample_descriptor_;
  metadata.reflection = Sample_reflection_;
  return metadata;
}

#if PROTOBUF_INLINE_NOT_IN_HEADERS
// Sample

// optional uint32 T = 1;
bool Sample::has_t() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
void Sample::set_has_t() {
  _has_bits_[0] |= 0x00000001u;
}
void Sample::clear_has_t() {
  _has_bits_[0] &= ~0x00000001u;
}
void Sample::clear_t() {
  t_ = 0u;
  clear_has_t();
}
::google::protobuf::uint32 Sample::t() const {
  // @@protoc_insertion_point(field_get:gps.Sample.T)
  return t_;
}
void Sample::set_t(::google::protobuf::uint32 value) {
  set_has_t();
  t_ = value;
  // @@protoc_insertion_point(field_set:gps.Sample.T)
}

// optional uint32 dX = 2;
bool Sample::has_dx() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
void Sample::set_has_dx() {
  _has_bits_[0] |= 0x00000002u;
}
void Sample::clear_has_dx() {
  _has_bits_[0] &= ~0x00000002u;
}
void Sample::clear_dx() {
  dx_ = 0u;
  clear_has_dx();
}
::google::protobuf::uint32 Sample::dx() const {
  // @@protoc_insertion_point(field_get:gps.Sample.dX)
  return dx_;
}
void Sample::set_dx(::google::protobuf::uint32 value) {
  set_has_dx();
  dx_ = value;
  // @@protoc_insertion_point(field_set:gps.Sample.dX)
}

// optional uint32 dU = 3;
bool Sample::has_du() const {
  return (_has_bits_[0] & 0x00000004u) != 0;
}
void Sample::set_has_du() {
  _has_bits_[0] |= 0x00000004u;
}
void Sample::clear_has_du() {
  _has_bits_[0] &= ~0x00000004u;
}
void Sample::clear_du() {
  du_ = 0u;
  clear_has_du();
}
::google::protobuf::uint32 Sample::du() const {
  // @@protoc_insertion_point(field_get:gps.Sample.dU)
  return du_;
}
void Sample::set_du(::google::protobuf::uint32 value) {
  set_has_du();
  du_ = value;
  // @@protoc_insertion_point(field_set:gps.Sample.dU)
}

// optional uint32 dO = 4;
bool Sample::has_do_() const {
  return (_has_bits_[0] & 0x00000008u) != 0;
}
void Sample::set_has_do_() {
  _has_bits_[0] |= 0x00000008u;
}
void Sample::clear_has_do_() {
  _has_bits_[0] &= ~0x00000008u;
}
void Sample::clear_do_() {
  do__ = 0u;
  clear_has_do_();
}
::google::protobuf::uint32 Sample::do_() const {
  // @@protoc_insertion_point(field_get:gps.Sample.dO)
  return do__;
}
void Sample::set_do_(::google::protobuf::uint32 value) {
  set_has_do_();
  do__ = value;
  // @@protoc_insertion_point(field_set:gps.Sample.dO)
}

// repeated float X = 5 [packed = true];
int Sample::x_size() const {
  return x_.size();
}
void Sample::clear_x() {
  x_.Clear();
}
float Sample::x(int index) const {
  // @@protoc_insertion_point(field_get:gps.Sample.X)
  return x_.Get(index);
}
void Sample::set_x(int index, float value) {
  x_.Set(index, value);
  // @@protoc_insertion_point(field_set:gps.Sample.X)
}
void Sample::add_x(float value) {
  x_.Add(value);
  // @@protoc_insertion_point(field_add:gps.Sample.X)
}
const ::google::protobuf::RepeatedField< float >&
Sample::x() const {
  // @@protoc_insertion_point(field_list:gps.Sample.X)
  return x_;
}
::google::protobuf::RepeatedField< float >*
Sample::mutable_x() {
  // @@protoc_insertion_point(field_mutable_list:gps.Sample.X)
  return &x_;
}

// repeated float U = 6 [packed = true];
int Sample::u_size() const {
  return u_.size();
}
void Sample::clear_u() {
  u_.Clear();
}
float Sample::u(int index) const {
  // @@protoc_insertion_point(field_get:gps.Sample.U)
  return u_.Get(index);
}
void Sample::set_u(int index, float value) {
  u_.Set(index, value);
  // @@protoc_insertion_point(field_set:gps.Sample.U)
}
void Sample::add_u(float value) {
  u_.Add(value);
  // @@protoc_insertion_point(field_add:gps.Sample.U)
}
const ::google::protobuf::RepeatedField< float >&
Sample::u() const {
  // @@protoc_insertion_point(field_list:gps.Sample.U)
  return u_;
}
::google::protobuf::RepeatedField< float >*
Sample::mutable_u() {
  // @@protoc_insertion_point(field_mutable_list:gps.Sample.U)
  return &u_;
}

// repeated float obs = 7 [packed = true];
int Sample::obs_size() const {
  return obs_.size();
}
void Sample::clear_obs() {
  obs_.Clear();
}
float Sample::obs(int index) const {
  // @@protoc_insertion_point(field_get:gps.Sample.obs)
  return obs_.Get(index);
}
void Sample::set_obs(int index, float value) {
  obs_.Set(index, value);
  // @@protoc_insertion_point(field_set:gps.Sample.obs)
}
void Sample::add_obs(float value) {
  obs_.Add(value);
  // @@protoc_insertion_point(field_add:gps.Sample.obs)
}
const ::google::protobuf::RepeatedField< float >&
Sample::obs() const {
  // @@protoc_insertion_point(field_list:gps.Sample.obs)
  return obs_;
}
::google::protobuf::RepeatedField< float >*
Sample::mutable_obs() {
  // @@protoc_insertion_point(field_mutable_list:gps.Sample.obs)
  return &obs_;
}

// repeated float meta = 8 [packed = true];
int Sample::meta_size() const {
  return meta_.size();
}
void Sample::clear_meta() {
  meta_.Clear();
}
float Sample::meta(int index) const {
  // @@protoc_insertion_point(field_get:gps.Sample.meta)
  return meta_.Get(index);
}
void Sample::set_meta(int index, float value) {
  meta_.Set(index, value);
  // @@protoc_insertion_point(field_set:gps.Sample.meta)
}
void Sample::add_meta(float value) {
  meta_.Add(value);
  // @@protoc_insertion_point(field_add:gps.Sample.meta)
}
const ::google::protobuf::RepeatedField< float >&
Sample::meta() const {
  // @@protoc_insertion_point(field_list:gps.Sample.meta)
  return meta_;
}
::google::protobuf::RepeatedField< float >*
Sample::mutable_meta() {
  // @@protoc_insertion_point(field_mutable_list:gps.Sample.meta)
  return &meta_;
}

inline const Sample* Sample::internal_default_instance() {
  return &Sample_default_instance_.get();
}
#endif  // PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

}  // namespace gps

// @@protoc_insertion_point(global_scope)